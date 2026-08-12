"""
Async SQLite session helpers for FootballVision.

All functions are async and use aiosqlite so they can be called
from FastAPI route handlers without blocking the event loop.
"""

import time
from pathlib import Path
from typing import Optional

try:
    import aiosqlite
    _AIOSQLITE = True
except ImportError:
    _AIOSQLITE = False

from db.models import ALL_TABLES

DB_PATH = Path(__file__).resolve().parent.parent / "footballvision.db"


async def init_db():
    """Create all tables if they do not exist."""
    if not _AIOSQLITE:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        for stmt in ALL_TABLES:
            await db.execute(stmt)
        await db.commit()


async def save_game(
    source: str = "screen",
    game_id: Optional[int] = None,
    ended: bool = False,
) -> int:
    """
    Insert or update a game row.

    Args:
        source:  Video source type string.
        game_id: If provided and ended=True, marks that game as finished.
        ended:   If True, set ended_at on the existing game_id row.

    Returns:
        The game id (new or existing).
    """
    if not _AIOSQLITE:
        return 0

    async with aiosqlite.connect(DB_PATH) as db:
        if ended and game_id:
            await db.execute(
                "UPDATE games SET ended_at = ? WHERE id = ?",
                (time.time(), game_id),
            )
            await db.commit()
            return game_id

        cursor = await db.execute(
            "INSERT INTO games (started_at, source_type) VALUES (?, ?)",
            (time.time(), source),
        )
        await db.commit()
        return cursor.lastrowid


async def save_play(
    game_id: int,
    formation: str,
    play_type: str,
    snap_frame: int,
    run_prob: float,
    pass_prob: float,
) -> int:
    """Insert a play row. Returns the new play id."""
    if not _AIOSQLITE:
        return 0

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO plays
               (game_id, timestamp, formation, play_type, snap_frame, run_prob, pass_prob)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (game_id, time.time(), formation, play_type, snap_frame, run_prob, pass_prob),
        )
        await db.commit()
        return cursor.lastrowid


async def save_prediction(
    play_id: int,
    session_id: str,
    prediction: str,
    correct: bool,
    points: int,
):
    """Insert a fan prediction row."""
    if not _AIOSQLITE:
        return

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO fan_predictions
               (play_id, session_id, prediction, correct, points)
               VALUES (?, ?, ?, ?, ?)""",
            (play_id, session_id, prediction, int(correct), points),
        )
        await db.commit()
