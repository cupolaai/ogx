#!/usr/bin/env python3
"""Write Cupola's image lock from the package metadata of a reference image.

Input: the JSON that dist_meta.py prints inside the reference image (name,
version and the wheel tags of every installed distribution). For each one,
the exact wheel file with those tags is looked up on PyPI (or, for a
`+cpu` version, the PyTorch CPU index) and written as
`name==version --hash=sha256:<that file>`, so pip installs exactly that file.
ogx and ogx-api are left out: they are built from this repository.
Fails when a distribution has no matching wheel on its index.
"""

import json
import re
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename

PYPI = "https://pypi.org/pypi/{name}/{version}/json"
TORCH_CPU = "https://download.pytorch.org/whl/cpu/{name}/"
SKIP = {"ogx", "ogx-api"}


def get(url):
    if not url.startswith("https://"):
        raise ValueError(f"only https indexes are read: {url}")
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - https only, checked above
        return r.read()


class Links(HTMLParser):
    """The wheel links (file name, sha256) of a PEP 503 simple index page."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            m = re.match(r"(?:.*/)?([^/#]+\.whl)#sha256=([0-9a-f]{64})", href)
            if m:
                self.links.append((urllib.parse.unquote(m.group(1)), m.group(2)))


def candidates(name, version):
    if "+cpu" in version:
        p = Links()
        p.feed(get(TORCH_CPU.format(name=canonicalize_name(name))).decode())
        return p.links
    data = json.loads(get(PYPI.format(name=name, version=version)))
    return [(u["filename"], u["digests"]["sha256"]) for u in data["urls"] if u["filename"].endswith(".whl")]


def main(meta_path):
    lines, missing = [], []
    for d in json.load(open(meta_path)):
        name, version = d["name"], d["version"]
        if canonicalize_name(name) in SKIP:
            continue
        want = set()
        for t in d["tags"]:
            want |= set(parse_tag(t))
        hit = None
        for filename, sha in candidates(name, version):
            _, fver, _, ftags = parse_wheel_filename(filename)
            if str(fver) == version and set(ftags) == want:
                hit = (filename, sha)
                break
        if hit is None:
            missing.append(f"{name}=={version} {sorted(str(t) for t in want)[:3]}")
            continue
        lines.append(f"{canonicalize_name(name)}=={version} --hash=sha256:{hit[1]}  # {hit[0]}")
    if missing:
        sys.exit("no matching wheel for:\n  " + "\n  ".join(missing))
    print("\n".join(sorted(lines)))


if __name__ == "__main__":
    main(sys.argv[1])
