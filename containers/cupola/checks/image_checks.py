#!/usr/bin/env python3
"""Build-time checks of Cupola's OGX image (run by containers/cupola/Containerfile).

1. ogx and ogx-api are this build's version.
2. Every requirement of the starter distribution (`ogx stack list-deps starter`)
   is satisfied by an installed package.
3. The milvus family (pymilvus, and what its milvus-lite and bulk_writer
   extras add) is installed at exactly the versions uv.lock resolves: the image installs from image-lock.txt only, uv.lock serves
   development and tests, and the two must not drift apart here.
4. The server imports; tiktoken's cl100k_base loads without network.
5. Python's SQLite is built with SECURE_DELETE and it is on: W9 relies on it so a
   deleted inference_store row is overwritten in the database file. A new base
   image without it must stop the build, not lose the guarantee unnoticed.
Any failure exits non-zero and stops the image build.
"""

import argparse
import re
import sqlite3
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, distribution, version

from packaging.requirements import InvalidRequirement, Requirement


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def closure(root, extras):
    seen, todo = {}, [(root, set(extras))]
    while todo:
        name, ex = todo.pop()
        try:
            d = distribution(name)
        except PackageNotFoundError:
            continue
        key = norm(d.metadata["Name"])
        if key in seen:
            continue
        seen[key] = d.version
        for r in d.requires or []:
            req = Requirement(r)
            if req.marker is None or any(req.marker.evaluate({"extra": e}) for e in (ex or {""})):
                todo.append((req.name, set(req.extras)))
    return seen


def sqlite_errors():
    """Python's SQLite must be built with SECURE_DELETE, and it must be on."""
    con = sqlite3.connect(":memory:")
    try:
        errors = []
        if "SECURE_DELETE" not in [row[0] for row in con.execute("PRAGMA compile_options")]:
            errors.append(f"SQLite {sqlite3.sqlite_version} is not built with SECURE_DELETE")
        if con.execute("PRAGMA secure_delete").fetchone()[0] != 1:
            errors.append(f"SQLite {sqlite3.sqlite_version}: PRAGMA secure_delete is not on by default")
        return errors
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uv-lock", required=True)
    ap.add_argument("--version", required=True)
    args = ap.parse_args()
    errors = []

    for pkg in ("ogx", "ogx-api"):
        if installed(pkg) != args.version:
            errors.append(f"{pkg} is {installed(pkg)}, want {args.version}")

    deps = subprocess.run(
        ["/usr/local/bin/ogx", "stack", "list-deps", "starter"], capture_output=True, text=True, check=True
    ).stdout
    checked = 0
    for token in deps.split():
        if token.startswith("-"):
            continue
        try:
            req = Requirement(token.strip("'\""))
        except InvalidRequirement:
            continue
        if req.marker is not None and not req.marker.evaluate():
            continue
        have = installed(req.name)
        checked += 1
        if have is None or (req.specifier and not req.specifier.contains(have, prereleases=True)):
            errors.append(f"starter requires {req}, installed {have}")
    if checked == 0:
        errors.append("ogx stack list-deps starter listed no requirement")

    lock = {norm(p["name"]): p.get("version") for p in tomllib.load(open(args.uv_lock, "rb"))["package"]}
    # The family: pymilvus and what its milvus-lite and bulk_writer extras add.
    # pymilvus's own dependencies (protobuf, numpy, grpcio, ...) are shared
    # with many packages and are not part of it.
    base = closure("pymilvus", set())
    family = {
        k: v for k, v in closure("pymilvus", {"milvus-lite", "bulk_writer"}).items() if k == "pymilvus" or k not in base
    }
    if "pymilvus" not in family or "minio" not in family:
        errors.append(f"milvus family incomplete: {sorted(family)}")
    for name, have in sorted(family.items()):
        if lock.get(name) != have:
            errors.append(f"milvus family: {name} installed {have}, uv.lock {lock.get(name)}")

    import tiktoken

    import ogx.core.server.server  # noqa: F401

    if len(tiktoken.get_encoding("cl100k_base").encode("Cupola")) == 0:
        errors.append("tiktoken cl100k_base did not encode")

    errors += sqlite_errors()

    if errors:
        print("cupola image checks FAILED:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    print(
        f"cupola image checks OK: ogx {args.version}; {checked} starter requirements satisfied; "
        f"milvus family ({len(family)} packages) equals uv.lock; tiktoken offline; "
        f"SQLite {sqlite3.sqlite_version} SECURE_DELETE on"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
