"""Train an outcome model on full and partial compositions for draft recommendations."""
import argparse
import json
from pathlib import Path

import torch
from sqlalchemy import select
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from database import get_engine, matches
from draft_features import DRAFT_FORMAT, partial_examples, role_evidence


class DraftOptimizerNN(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc3(self.relu(self.fc2(x)))


def validate_rows(rows):
    if len(rows) < 10:
        raise ValueError('Collect at least 10 matches for a smoke test; use many more for useful training.')
    seen = set()
    for row in rows:
        if row['match_id'] in seen:
            raise ValueError('Duplicate match IDs in training snapshot.')
        seen.add(row['match_id'])
        ids = row['team_1'] + row['team_2']
        if (len(row['team_1']) != 5 or len(row['team_2']) != 5 or len(set(ids)) != 10
                or any(type(c) is not int or c <= 0 for c in ids)):
            raise ValueError('Each match needs ten distinct positive champion IDs.')
        if type(row['team_1_win']) is not bool or row['queue_id'] != 420:
            raise ValueError('Expected ranked solo matches with boolean outcomes.')


def encode_matches(rows, champion_to_index):
    size = len(champion_to_index)
    features = torch.zeros((len(rows), size * 2), dtype=torch.float32)
    labels = torch.tensor([r['team_1_win'] for r in rows], dtype=torch.float32).unsqueeze(1)
    for i, row in enumerate(rows):
        for side, team in enumerate(('team_1', 'team_2')):
            for champion_id in row[team]:
                features[i, side * size + champion_to_index[champion_id]] = 1
    return features, labels


def train_model(rows, output_dir, epochs=10, batch_size=32, seed=42, val_fraction=0.2,
                partial_drafts=True):
    if epochs < 1 or batch_size < 1 or not 0 < val_fraction < 1:
        raise ValueError('Use positive epochs/batch size and 0 < val_fraction < 1.')
    validate_rows(rows)
    # Split original matches before any future augmentation; newest games validate.
    rows = sorted(rows, key=lambda r: (r['game_creation'], r['match_id']))
    split = max(1, min(len(rows) - 1, int(len(rows) * (1 - val_fraction))))
    # Vocabulary only: no outcome statistics are calculated from validation games.
    champion_ids = sorted({c for r in rows for team in ('team_1', 'team_2') for c in r[team]})
    champion_to_index = {c: i for i, c in enumerate(champion_ids)}
    X, y = encode_matches(rows, champion_to_index)
    train_X, train_y = X[:split], y[:split]
    val_X, val_y = X[split:], y[split:]
    roles = role_evidence(rows[:split]) if partial_drafts else {}
    if partial_drafts:
        if not roles:
            raise ValueError('No usable role labels in training matches; collect role data first.')
        train_X, train_y = partial_examples(train_X, train_y, seed)
        val_X, val_y = partial_examples(val_X, val_y, seed + 1)
    torch.manual_seed(seed)
    model = DraftOptimizerNN(X.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'dataset.json').write_text(json.dumps(rows, indent=2) + '\n')
    baseline_probability = y[:split].mean().clamp(1e-6, 1 - 1e-6)
    baseline_loss = nn.functional.binary_cross_entropy(
        baseline_probability.expand_as(val_y), val_y).item()
    baseline_accuracy = ((baseline_probability >= 0.5) == val_y.bool()).float().mean().item()
    print(f'Snapshot: {len(rows)} matches, {split} train, {len(rows)-split} validation; '
          f'{len(champion_ids)} champions', flush=True)
    print(f'Constant baseline: validation_loss={baseline_loss:.4f}, accuracy={baseline_accuracy:.3f}', flush=True)
    history = []
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_X)
        model.eval()
        with torch.no_grad():
            logits = model(val_X)
            validation_loss = criterion(logits, val_y).item()
            accuracy = ((logits >= 0) == val_y.bool()).float().mean().item()
            full_loss = criterion(model(X[split:]), y[split:]).item()
            partial_loss = criterion(logits[len(rows)-split:], val_y[len(rows)-split:]).item() if partial_drafts else None
        metrics = {'epoch': epoch, 'train_loss': total_loss / len(train_X),
                   'full_validation_loss': full_loss, 'partial_validation_loss': partial_loss,
                   'validation_loss': validation_loss, 'validation_accuracy': accuracy}
        history.append(metrics)
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': X.shape[1], 'output_dim': 1, 'hidden_dim': 256,
                'champion_to_index': champion_to_index,
                'trained_champion_ids': sorted({c for r in rows[:split] for t in ('team_1', 'team_2') for c in r[t]}),
                'seed': seed, 'metrics': metrics, 'target': 'team_1_win',
                'format': DRAFT_FORMAT if partial_drafts else 'full_composition_v1',
                'role_counts': roles,
                'train_match_ids': [r['match_id'] for r in rows[:split]],
                'validation_match_ids': [r['match_id'] for r in rows[split:]],
            }, output_dir / 'model.pt')
        (output_dir / 'metrics.json').write_text(json.dumps({
            'baseline_validation_loss': baseline_loss,
            'baseline_validation_accuracy': baseline_accuracy, 'history': history,
        }, indent=2) + '\n')
        print(f'Epoch {epoch}/{epochs}: train_loss={total_loss / len(train_X):.4f} '
              f'validation_loss={validation_loss:.4f} accuracy={accuracy:.3f}', flush=True)
    print(f"Best checkpoint: {output_dir / 'model.pt'}", flush=True)
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, help='Reuse a saved dataset.json instead of querying PostgreSQL')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/draft'))
    parser.add_argument('--full-compositions-only', action='store_true', help='Train the old baseline; not usable by the recommendation API')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2, help='CPU threads; leaves capacity for other work')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error('Epochs, batch size, and threads must be positive.')
    torch.set_num_threads(args.threads)
    try:
        if args.dataset:
            rows = json.loads(args.dataset.read_text())
        else:
            with get_engine().connect() as connection:
                rows = [dict(row) for row in connection.execute(select(matches)).mappings()]
        train_model(rows, args.output_dir, args.epochs, args.batch_size, args.seed,
                    partial_drafts=not args.full_compositions_only)
    except ValueError as exc:
        parser.exit(1, f'{exc}\n')
    except Exception as exc:
        parser.exit(1, f'Training failed ({type(exc).__name__}). Check the database/schema or dataset and output path.\n')
