"""Rows left in inference_store and how often the seed marker occurs in the database files.

Runs inside the OGX image: python3 -I inspect_inference_store.py <sql_store.db>
"""

import os
import sqlite3
import sys

db = sys.argv[1]
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
rows = con.execute("SELECT COUNT(*) FROM inference_store").fetchone()[0]
con.close()
found = {}
for path in (db, db + "-wal", db + "-journal"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            found[os.path.basename(path)] = (os.path.getsize(path), f.read().count(b"CUPOLA-PURGE-MARKER"))
files = ", ".join(f"{n} {s} bytes marker x{c}" for n, (s, c) in found.items())
print(f"rows {rows}; {files}")
