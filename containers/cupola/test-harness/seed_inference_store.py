"""Fill inference_store with copies of the one row OGX wrote, each carrying a marker.

Runs inside the OGX image: python3 -I seed_inference_store.py <sql_store.db> <rows> <bytes per row>
"""

import json
import sqlite3
import sys

MARKER = "CUPOLA-PURGE-MARKER"
db, rows, size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
con = sqlite3.connect(db)
con.row_factory = sqlite3.Row
first = con.execute("SELECT * FROM inference_store").fetchall()
if len(first) != 1:
    sys.exit(f"expected the one row OGX wrote, found {len(first)}")
row = dict(first[0])
cols = list(row)
pad = (MARKER + " ") * (size // (len(MARKER) + 1))
batch = []
for i in range(1, rows):
    r = dict(row, id=f"chatcmpl-seed-{i}", created=row["created"] + i)
    msgs = json.loads(r["input_messages"])
    msgs[0]["content"] = f"{MARKER} {i} {pad}"
    r["input_messages"] = json.dumps(msgs)
    batch.append(tuple(r[c] for c in cols))
# The column names come from the table OGX created, the values are bound parameters.
sql = f"INSERT INTO inference_store ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"  # noqa: S608
con.executemany(sql, batch)
con.commit()
print("rows", con.execute("SELECT COUNT(*) FROM inference_store").fetchone()[0])
con.close()
