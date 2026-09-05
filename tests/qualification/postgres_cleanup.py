"""Parent-owned disposable PostgreSQL lifecycle using the selected interpreter."""

import subprocess

DATABASE_ENV = "CAYU_QUALIFICATION_DATABASE"

# Use the candidate's interpreter for psycopg; the CLI itself needs only stdlib.
_OPERATION = """
import os, sys
import psycopg
from psycopg import sql
operation, database = sys.argv[1:]
with psycopg.connect(
    os.environ['CAYU_TEST_POSTGRES_DSN'], autocommit=True, connect_timeout=10,
    options='-c statement_timeout=10000 -c lock_timeout=10000',
) as connection:
    # Serialize cleanup with any CREATE whose acknowledgement was lost.
    connection.execute('SELECT pg_advisory_lock(hashtextextended(%s, 0))', (database,))
    statement = 'CREATE DATABASE {}' if operation == 'create' else 'DROP DATABASE IF EXISTS {} WITH (FORCE)'
    connection.execute(sql.SQL(statement).format(sql.Identifier(database)))
    if operation == 'drop':
        assert connection.execute('SELECT 1 FROM pg_database WHERE datname = %s', (database,)).fetchone() is None
"""


def postgres_operation(python, database, env, operation):
    try:
        result = subprocess.run(
            [python, "-c", _OPERATION, operation, database],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
