"""Rank available champions by scoring each candidate in the submitted draft."""
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from draft_features import DRAFT_FORMAT
from draft_optimizer import DraftOptimizerNN

ChampionId = Annotated[int, Field(strict=True, gt=0)]


class DraftRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    allies: list[ChampionId] = Field(default_factory=list, max_length=4)
    enemies: list[ChampionId] = Field(default_factory=list, max_length=5)
    bans: list[ChampionId] = Field(default_factory=list, max_length=10)
    role: Literal['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
    side: Literal['blue', 'red'] = 'blue'
    top_k: int = Field(default=5, ge=1, le=20)
    min_role_games: int = Field(default=5, ge=1)

    @model_validator(mode='after')
    def unique_champions(self):
        ids = self.allies + self.enemies + self.bans
        if len(ids) != len(set(ids)):
            raise ValueError('Picks and bans must be distinct; a champion cannot appear twice.')
        return self


class Recommender:
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint.get('format') != DRAFT_FORMAT:
            raise ValueError('Retrain with partial-draft training; this checkpoint is a full-team baseline.')
        self.mapping = checkpoint['champion_to_index']
        self.supported = set(checkpoint['trained_champion_ids'])
        self.role_counts = checkpoint['role_counts']
        self.metrics = checkpoint['metrics']
        self.training_matches = len(checkpoint['train_match_ids'])
        self.model = DraftOptimizerNN(checkpoint['input_dim'], checkpoint['output_dim'], checkpoint['hidden_dim'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def champions(self):
        return [{'champion_id': champion, 'role_games': self.role_counts.get(champion, {})}
                for champion in sorted(self.supported)]

    def recommend(self, draft: DraftRequest):
        unknown = set(draft.allies + draft.enemies) - self.supported
        if unknown:
            raise ValueError(f'Picks lack training coverage: {sorted(unknown)}. Collect more matches and retrain.')
        excluded = set(draft.allies + draft.enemies + draft.bans)
        candidates = sorted(champion for champion in self.supported - excluded
                            if self.role_counts.get(champion, {}).get(draft.role, 0) >= draft.min_role_games)
        metadata = {
            'score_type': 'uncalibrated_ally_win_estimate',
            'training_matches': self.training_matches,
            'eligible_candidates': len(candidates),
            'notes': ['Experimental ranking from synthetic partial drafts; scores are not proven pick improvements.',
                      'Role filters use training history. Rank conditioning and automated client integration are not implemented.'],
        }
        if not candidates:
            return {**metadata, 'recommendations': [],
                    'message': 'No available champions meet the role-history threshold. Collect more data or lower min_role_games.'}
        size = len(self.mapping)
        features = torch.zeros((len(candidates), size * 2), dtype=torch.float32)
        ally_offset, enemy_offset = (0, size) if draft.side == 'blue' else (size, 0)
        for champion in draft.allies:
            features[:, self.mapping[champion] + ally_offset] = 1
        for champion in draft.enemies:
            features[:, self.mapping[champion] + enemy_offset] = 1
        for row, champion in enumerate(candidates):
            features[row, self.mapping[champion] + ally_offset] = 1
        # One batched forward pass for the whole candidate pool.
        with torch.inference_mode():
            blue_scores = torch.sigmoid(self.model(features)).flatten()
            scores = blue_scores if draft.side == 'blue' else 1 - blue_scores
        ranked = sorted(zip(candidates, scores.tolist()), key=lambda item: (-item[1], item[0]))
        recommendations = [
            {'champion_id': champion, 'score': score,
             'role_games': self.role_counts[champion][draft.role],
             'reason': f'Available pick with recorded {draft.role} games; ranked by the current composition model.'}
            for champion, score in ranked[:draft.top_k]
        ]
        return {**metadata, 'recommendations': recommendations}
