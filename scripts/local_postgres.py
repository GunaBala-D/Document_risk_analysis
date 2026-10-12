"""Start a local PostgreSQL server without Docker (development convenience).

Uses the pip-installable `pgserver` package (real PostgreSQL binaries). In docker-compose
the `postgres` service is used instead.

Usage:
    python -m scripts.local_postgres              # start (or reuse) the server, print DATABASE_URL
    python -m scripts.local_postgres --write-env  # ...and update DATABASE_URL in .env

The server stops when the process that started it exits, and may come back on a different
port, so --write-env keeps .env in sync. Keep the process running while using the API.
"""

from __future__ import annotations

import argparse
import re
import time

from src.config import DATA_DIR, PROJECT_ROOT


def start(data_dir=None):
    import pgserver

    data_dir = data_dir or DATA_DIR / "pgdata"
    data_dir.mkdir(parents=True, exist_ok=True)
    return pgserver.get_server(str(data_dir), cleanup_mode=None)


def database_url(server, dbname: str = "document_risk") -> str:
    exists = server.psql(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}';")
    if "1 row" not in exists:
        server.psql(f"CREATE DATABASE {dbname};")
    uri = server.get_uri(dbname)
    return uri.replace("postgresql://", "postgresql+psycopg://", 1)


def write_env(url: str) -> None:
    env = PROJECT_ROOT / ".env"
    text = env.read_text(encoding="utf-8-sig") if env.exists() else ""  # tolerate a BOM (PowerShell-written files)
    line = f"DATABASE_URL={url}"
    text = re.sub(r"(?m)^DATABASE_URL=.*$", line, text) if "DATABASE_URL=" in text else f"{line}\n{text}"
    env.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-env", action="store_true")
    ap.add_argument("--keep-alive", action="store_true", help="block so the server stays up")
    args = ap.parse_args()
    srv = start()
    url = database_url(srv)
    if args.write_env:
        write_env(url)
    print(url, flush=True)
    while args.keep_alive:
        time.sleep(3600)
