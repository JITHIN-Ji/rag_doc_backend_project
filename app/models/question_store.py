import aiosqlite
from typing import List, Dict
from pathlib import Path
from typing import List, Dict, Optional 

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "chat_history.db" 
MAX_ROWS = 10


CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT,
    google_id TEXT UNIQUE,
    is_active BOOLEAN DEFAULT 1
);
"""

CREATE_QUESTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS questions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  TEXT NOT NULL,
    query    TEXT,
    created  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


async def _get_conn():
    conn = await aiosqlite.connect(DB_PATH)
    await conn.execute("PRAGMA foreign_keys = ON;") 
    await conn.execute(CREATE_USERS_TABLE_SQL)
    await conn.execute(CREATE_QUESTIONS_TABLE_SQL)
    await conn.commit()
    return conn


async def save_query(user_id: str, query: str):
    conn = await _get_conn()
    await conn.execute(
        "INSERT INTO questions (user_id, query) VALUES (?, ?)",
        (user_id, query),
    )
    await conn.execute(
        """
        DELETE FROM questions
        WHERE id NOT IN (
            SELECT id FROM questions
            WHERE user_id = ?
            ORDER BY created DESC
            LIMIT ?
        )
        AND user_id = ?;
        """,
        (user_id, MAX_ROWS, user_id),
    )
    await conn.commit()
    await conn.close()

async def get_recent(user_id: str, limit: int = MAX_ROWS) -> List[Dict[str, str]]:
    conn = await _get_conn()
    cur = await conn.execute(
        """
        SELECT query, created
        FROM questions
        WHERE user_id = ?
        ORDER BY created DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = await cur.fetchall()
    await conn.close()
    return [{"query": r[0], "created": r[1]} for r in rows]



async def get_user_by_email(email: str) -> Optional[Dict]:
    conn = await _get_conn()
    conn.row_factory = aiosqlite.Row 
    cursor = await conn.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = await cursor.fetchone()
    await conn.close()
    return dict(user) if user else None

async def create_user(email: str, hashed_password: str) -> Dict:
    conn = await _get_conn()
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute(
        "INSERT INTO users (email, hashed_password) VALUES (?, ?)",
        (email, hashed_password)
    )
    await conn.commit()
    user_id = cursor.lastrowid
    
    
    cursor = await conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    new_user = await cursor.fetchone()
    await conn.close()
    return dict(new_user)