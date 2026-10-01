"""Partial-composition augmentation and training-only role evidence."""
import torch

ROLES = ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')
DRAFT_FORMAT = 'partial_composition_v1'


def partial_examples(features, labels, seed, variants=8):
    """Keep each full game and add fixed random subsets of its selected champions.

    Call separately AFTER splitting original matches. These are synthetic partial
    compositions, not recovered historical pick orders.
    """
    generator = torch.Generator().manual_seed(seed)
    size = features.shape[1] // 2
    examples = [features]
    for _ in range(variants):
        masked = torch.zeros_like(features)
        for index, row in enumerate(features):
            counts = torch.randint(0, 6, (2,), generator=generator).tolist()
            if counts == [0, 0]:
                counts[0] = 1
            for side, count in enumerate(counts):
                selected = row[side * size:(side + 1) * size].nonzero().flatten()
                keep = selected[torch.randperm(len(selected), generator=generator)[:count]]
                masked[index, keep + side * size] = 1
        examples.append(masked)
    return torch.cat(examples), labels.repeat(variants + 1, 1)


def role_evidence(rows):
    result = {}
    for row in rows:
        for team in ('team_1', 'team_2'):
            roles = row.get(f'{team}_roles', [])
            if len(roles) != 5:
                raise ValueError('Partial-draft training requires five recorded roles per team.')
            # Ambiguous/absent Riot role labels do not establish role eligibility.
            for champion, role in zip(row[team], roles):
                if role in ROLES:
                    counts = result.setdefault(champion, {})
                    counts[role] = counts.get(role, 0) + 1
    return result
