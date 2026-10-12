# Cupola's OGX image

`cupola-cr.io/cupola/cupola-distribution-starter:1.1.3-cupola.N` is built from this fork, branch `cupola/1.1.3`
(upstream tag `v1.1.3` plus Cupola's source commits). Nothing in the image patches third-party code at build time:
every change is a commit here.

```sh
containers/cupola/build.sh 1.1.3-cupola.9           # local image
containers/cupola/build.sh 1.1.3-cupola.9 --push    # to cupola-cr.io
```

## The only source of the image's Python packages: `image-lock.txt`

- `image-lock.txt` pins every third-party package to **one exact wheel file by sha256**. The image installs exactly
  those files — `pip install --no-index --no-deps --require-hashes` from a wheelhouse — and nothing else. No resolver
  runs and no package index is reached while the image builds (`RUN --network=none`).
- `ogx` and `ogx-api` are not in it: `build.sh` builds them from the committed source, sdist then wheel, with
  setuptools 80.10.2 (`build-constraints.txt`) — the way PyPI's 1.1.3 wheels were built; a build of tag `v1.1.3` this
  way is byte-identical to them. Version `1.1.3+cupola.N`.
- `uv.lock` installs **nothing** into the image; it serves development and tests. Its **milvus family** — pymilvus,
  everything its `milvus-lite` and `bulk_writer` extras pull in — must equal the image's versions:
  `checks/image_checks.py` compares them inside the build and stops it on any difference.
- The lock was generated from Cupola's live image `1.1.3-cupola.5` (the header of `image-lock.txt` has the commands):
  the same 238 third-party packages, file for file, except four set back to the official 1.1.3 versions on purpose
  (protobuf, grpcio, numpy, typing-extensions — cupola.5 got newer ones by accident and failed `pip check`; the
  header says why).
- Every module is compiled to bytecode in the image (`compileall`), so a pod start does not compile them again.

## The rest of the image

Follows upstream's `containers/Containerfile` at `v1.1.3` (the recipe of `docker.io/ogxai/distribution-starter:1.1.3`):
the base is `python:3.12.13-slim-trixie` of 2026-06-24 **by digest** (the official image's four base layers;
docker-library/repo-info `82a0a1993672`), upstream's apt packages come from the Debian snapshot of the official build
(`APT_SNAPSHOT`), and the environment, entrypoint and directories are upstream's. tiktoken's `cl100k_base` is in the
image (checked by sha256), so the server never downloads it.

## Build steps and limits

`build.sh` needs a clean working tree, docker buildx, and network only for its download steps: (1) the ogx wheels,
(2) the wheelhouse — downloaded natively, the PyTorch CPU index for the `+cpu` packages, the file count checked
against the lock, (3) tiktoken, (4) the `linux/amd64` image, timestamped, stopped after 45 minutes. The build-time
checks (`pip check`, the starter distribution's requirements, the milvus family against `uv.lock`, imports, tiktoken
offline) stop the build on any failure.

## Air-gapped clusters

The image is complete: a cluster without internet needs only this one image in its mirror. Building it needs the
network (PyPI, the PyTorch CPU index, snapshot.debian.org, the tiktoken file); the build host is not the cluster.
