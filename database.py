"""Shared connection settings and match storage schema."""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import BigInteger, Boolean, Column, Integer, JSON, MetaData, String, Table, create_engine

load_dotenv(Path(__file__).with_name('.env'))
metadata = MetaData()
matches = Table(
    'matches', metadata,
    Column('match_id', String, primary_key=True),
    Column('team_1', JSON, nullable=False),
    Column('team_2', JSON, nullable=False),
    Column('team_1_roles', JSON, nullable=False),
    Column('team_2_roles', JSON, nullable=False),
    Column('team_1_win', Boolean, nullable=False),
    Column('game_version', String, nullable=False),
    Column('game_creation', BigInteger, nullable=False),
    Column('queue_id', Integer, nullable=False),
)


def get_engine():
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError('Set DATABASE_URL in .env (see .env.example).')
    engine = create_engine(db_url, pool_pre_ping=True, hide_parameters=True)
    if engine.dialect.name != 'postgresql':
        raise ValueError('DATABASE_URL must point to PostgreSQL.')
    return engine
