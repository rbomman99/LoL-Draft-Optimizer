"""Collect ranked solo matches: python data-pipeline.py --max-matches 500."""
import argparse
import logging
import os
import time

import requests
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from database import get_engine, matches, metadata

LOG = logging.getLogger(__name__)


class RiotClient:
    def __init__(self):
        api_key = os.getenv('RIOT_API_KEY')
        if not api_key or api_key == 'replace_with_your_riot_api_key':
            raise ValueError('Set RIOT_API_KEY in .env before collecting matches.')
        self.platform = os.getenv('RIOT_PLATFORM', 'euw1')
        self.region = os.getenv('RIOT_REGION', 'europe')
        self.session = requests.Session()
        self.session.headers.update({'X-Riot-Token': api_key})

    def get(self, route, path, params=None):
        url = f'https://{route}.api.riotgames.com{path}'
        for attempt in range(6):
            # Conservative pacing; server Retry-After remains authoritative.
            time.sleep(1.3)
            try:
                response = self.session.get(url, params=params, timeout=30)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 5:
                    raise RuntimeError('Riot API network retries exhausted.') from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 5:
                    raise RuntimeError(f'Riot API retries exhausted (HTTP {response.status_code}).')
                delay = max(float(response.headers.get('Retry-After', 2 ** attempt)), 1)
                LOG.warning('Riot HTTP %s; retrying in %.0fs', response.status_code, delay)
                time.sleep(delay)
                continue
            if response.status_code in (401, 403):
                raise RuntimeError('Riot rejected the API key; check or renew RIOT_API_KEY.')
            response.raise_for_status()
            return response.json()
        raise RuntimeError('Riot request failed.')

    def get_challenger_summoners(self):
        data = self.get(self.platform, '/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_DUO_5x5')
        entries = data.get('entries', [])
        if any('puuid' not in entry for entry in entries):
            raise ValueError('League entries are missing PUUIDs; check the Riot API response schema.')
        return [entry['puuid'] for entry in entries]

    def get_match_ids(self, puuid, count=100):
        for start in range(0, count, 100):
            size = min(100, count - start)
            ids = self.get(self.region, f'/lol/match/v5/matches/by-puuid/{puuid}/ids',
                           {'queue': 420, 'start': start, 'count': size})
            yield from ids
            if len(ids) < size:
                break


def parse_match(data):
    info = data['info']
    if info.get('queueId') != 420 or info.get('gameDuration', 0) < 300:
        return None
    participants = info['participants']
    if any(p.get('gameEndedInEarlySurrender', False) for p in participants):
        return None
    blue = [p for p in participants if p['teamId'] == 100]
    red = [p for p in participants if p['teamId'] == 200]
    if len(participants) != 10 or len(blue) != 5 or len(red) != 5:
        raise ValueError('Expected two teams of five participants.')
    champion_ids = [p['championId'] for p in participants]
    if len(set(champion_ids)) != 10 or any(c <= 0 for c in champion_ids):
        raise ValueError('Invalid ranked champion composition.')
    if (any(p['win'] != blue[0]['win'] for p in blue)
            or any(p['win'] == blue[0]['win'] for p in red)):
        raise ValueError('Inconsistent team outcomes.')
    return {
        'match_id': data['metadata']['matchId'],
        'team_1': [p['championId'] for p in blue],
        'team_2': [p['championId'] for p in red],
        'team_1_roles': [p.get('teamPosition', '') for p in blue],
        'team_2_roles': [p.get('teamPosition', '') for p in red],
        'team_1_win': blue[0]['win'],
        'game_version': info['gameVersion'],
        'game_creation': info['gameCreation'],
        'queue_id': info['queueId'],
    }


def ingestion_pipeline(max_matches=500, matches_per_player=100):
    engine = get_engine()
    client = RiotClient()
    metadata.create_all(engine)
    with engine.connect() as connection:
        visited = set(connection.execute(select(matches.c.match_id)).scalars())
    added = 0
    for puuid in client.get_challenger_summoners():
        for match_id in client.get_match_ids(puuid, matches_per_player):
            if match_id in visited:
                continue
            data = client.get(client.region, f'/lol/match/v5/matches/{match_id}')
            row = parse_match(data)
            if row is None:
                visited.add(match_id)
                continue
            # Commit each match so interruptions preserve progress.
            with engine.begin() as connection:
                result = connection.execute(insert(matches).values(**row).on_conflict_do_nothing(
                    index_elements=[matches.c.match_id]))
            visited.add(match_id)
            added += result.rowcount
            LOG.info('Stored %s (%d new matches this run)', match_id, added)
            if added >= max_matches:
                return added
    LOG.info('Finished available histories; stored %d new matches.', added)
    return added


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-matches', type=int, default=500, help='Maximum NEW matches this run')
    parser.add_argument('--matches-per-player', type=int, default=100)
    args = parser.parse_args()
    if args.max_matches < 1 or args.matches_per_player < 1:
        parser.error('Match limits must be positive.')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
        ingestion_pipeline(args.max_matches, args.matches_per_player)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')
    except Exception as exc:
        # Avoid exposing connection details from database exception strings.
        parser.exit(1, f'Collection failed ({type(exc).__name__}). Check DB connectivity/schema and Riot service availability.\n')
