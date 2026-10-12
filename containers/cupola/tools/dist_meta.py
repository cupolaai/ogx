#!/usr/bin/env python3
"""Print every installed distribution's name, version and wheel tags as JSON.

Run inside a reference image (python3 dist_meta.py > meta.json); its output
is the input of make_image_lock.py.
"""

import json
from importlib.metadata import distributions
from pathlib import Path


def wheel_info(dist_path: Path) -> tuple[list[str], str]:
    tags, installer = [], ""
    wheel = dist_path / "WHEEL"
    if wheel.exists():
        for line in wheel.read_text().splitlines():
            if line.startswith("Tag:"):
                tags.append(line.split(":", 1)[1].strip())
    inst = dist_path / "INSTALLER"
    if inst.exists():
        installer = inst.read_text().strip()
    return tags, installer


def main() -> None:
    out = []
    for d in distributions():
        path = getattr(d, "_path", None)
        tags, installer = wheel_info(Path(path)) if path else ([], "")
        out.append({"name": d.metadata["Name"], "version": d.version, "tags": tags, "installer": installer})
    print(json.dumps(sorted(out, key=lambda x: x["name"].lower()), indent=0))


if __name__ == "__main__":
    main()
