# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: the security-pinned packages are present, new enough, and still work.

Cupola's scan of 2026-10-30 (trivy 0.70.0 against the running image
1.1.3-cupola.17) found eight packages below the version that closes their
CVEs — 12 CRITICAL and 90 HIGH in all. image-lock.txt was raised to the
versions the SCANNER names (not the ones upstream's pyproject comments name:
those were measured to be wrong for pillow and authlib).

The jump that can actually break something is cryptography 49 -> 50, which is
a major release: it raises its own floor to cffi>=2.0.0 and is constrained
from above by msal (cryptography<51). Nothing in OGX imports cryptography
directly — it arrives through Authlib, joserfc, msal, google-auth,
pdfminer.six and the azure clients — so a break would not surface as an
ImportError in our code but as a failure deep inside one of those, at
runtime, on a customer's cluster. Hence this test: it exercises the primitives
those libraries actually use.

What each check is for:
  - version floors: a future regeneration of image-lock.txt must not silently
    drop back below a CVE fix.
  - cryptography: x509 parsing + RSA/EC sign-verify + PKCS#7-adjacent
    serialisation are the paths msal/azure-identity/joserfc drive. CVE-2026-69247
    was a PKCS#7 oracle, so the serialisation path is deliberately included.
  - pyjwt + cryptography together: ogx pins pyjwt[crypto], i.e. RS256. An
    algorithm-confusion fix landed upstream, so a forged-token rejection is
    asserted rather than just a happy-path round trip.
  - Authlib 1.9.1 pulls in joserfc>=1.6.8 (already pinned at 1.7.1). If that
    pairing is wrong the failure is an ImportError at first use, which is
    exactly what the import here catches.
  - pillow 13: the C extension (PIL._imaging) must load and decode. pypdf
    imports PIL eagerly, so a broken wheel would break PDF ingestion even
    though Cupola never calls the image paths.
"""

import importlib.metadata as md

import pytest
from packaging.version import Version

# package -> the minimum that closes the CVEs trivy reported (2026-10-30)
PINNED_FLOORS = {
    "cryptography": "50.0.0",
    "Authlib": "1.9.1",
    "pyjwt": "2.14.0",
    "pillow": "13.0.0",
    "nltk": "3.10.4",
    "aiohttp": "3.14.4",
    "pymongo": "4.18.3",
    "pyasn1": "0.6.4",
    # S-7c (2026-11-02): both CVEs were published AFTER the 2026-10-30 patch,
    # so cupola.18 still shipped them. anyio carries CVE-2026-63374 below
    # 4.14.2; certifi carries GHSA-6q48-6c7w-9gw2 below 2026.10.18 (a
    # compromised root staying in the trust store — it reaches every outbound
    # TLS call OGX makes, which is why it is pinned rather than accepted).
    "anyio": "4.14.2",
    "certifi": "2026.10.18",
}


@pytest.mark.parametrize("name,floor", sorted(PINNED_FLOORS.items()))
def test_security_pin_is_at_or_above_its_cve_floor(name: str, floor: str) -> None:
    try:
        have = md.version(name)
    except md.PackageNotFoundError:  # pragma: no cover - would mean the image lost a package
        pytest.fail(f"{name} is not installed; image-lock.txt pins it")
    assert Version(have) >= Version(floor), (
        f"{name} {have} is below {floor}, the version trivy names as the CVE fix. "
        "Raising it in containers/cupola/image-lock.txt is the only supported way."
    )


def test_cryptography_parses_a_certificate_and_signs_both_key_types() -> None:
    """cryptography 50: x509 + RSA/EC sign-verify, the paths msal and azure drive."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    from cryptography.x509.oid import NameOID

    # EC sign + verify
    ec_key = ec.generate_private_key(ec.SECP256R1())
    sig = ec_key.sign(b"cupola", ec.ECDSA(hashes.SHA256()))
    ec_key.public_key().verify(sig, b"cupola", ec.ECDSA(hashes.SHA256()))

    # RSA sign + verify (PKCS1v15 is what the token paths use)
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rsig = rsa_key.sign(b"cupola", padding.PKCS1v15(), hashes.SHA256())
    rsa_key.public_key().verify(rsig, b"cupola", padding.PKCS1v15(), hashes.SHA256())

    # Self-signed certificate: build, serialise, parse back.
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "cupola.test")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(rsa_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(__import__("datetime").datetime(2020, 1, 1))
        .not_valid_after(__import__("datetime").datetime(2040, 1, 1))
        .sign(rsa_key, hashes.SHA256())
    )
    pem = cert.public_bytes(serialization.Encoding.PEM)
    assert x509.load_pem_x509_certificate(pem).subject == name

    # DER round trip too: the serialisation path CVE-2026-69247 touched.
    der = cert.public_bytes(serialization.Encoding.DER)
    assert x509.load_der_x509_certificate(der).subject == name


def test_pyjwt_rs256_round_trip_and_rejects_a_forged_token() -> None:
    """ogx pins pyjwt[crypto]; RS256 must work AND a tampered token must fail."""
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    token = jwt.encode({"sub": "cupola"}, priv, algorithm="RS256")
    assert jwt.decode(token, pub, algorithms=["RS256"])["sub"] == "cupola"

    # Algorithm confusion: the classic attack signs HS256 with the RSA PUBLIC
    # key, because a verifier that trusts the token's own `alg` would then
    # accept it. pyjwt 2.14 refuses this at BOTH ends; measured here (the first
    # version of this test only asserted the decode side and failed, because
    # encode raises first — the stronger behaviour).
    with pytest.raises(jwt.InvalidKeyError):
        jwt.encode({"sub": "attacker"}, pub, algorithm="HS256")

    # And a token that really is HS256 must not be accepted as RS256, even
    # when the caller passes the right key material.
    hs = jwt.encode({"sub": "attacker"}, "a-test-only-secret-of-32-bytes!!", algorithm="HS256")
    with pytest.raises(jwt.InvalidAlgorithmError):
        jwt.decode(hs, pub, algorithms=["RS256"])


def test_authlib_and_its_joserfc_dependency_import_and_sign() -> None:
    """Authlib 1.9.1 requires joserfc>=1.6.8; a wrong pairing fails at import.

    Authlib 1.9 deprecates its own `authlib.jose` in favour of joserfc, so the
    signing round trip is driven through joserfc — that is the code path the
    new Authlib actually relies on, and importing Authlib proves the pair
    resolves.
    """
    import authlib
    from joserfc import jws
    from joserfc.jwk import OctKey

    assert Version(authlib.__version__) >= Version("1.9.1")

    # Build the key with joserfc's own constructor: a hand-written JWK dict is
    # easy to get wrong (the first version of this test did, and failed in
    # joserfc/jwk.py rather than in anything Cupola ships).
    key = OctKey.import_key("a-test-only-key-of-32-bytes-len!")
    token = jws.serialize_compact({"alg": "HS256"}, b"cupola", key)
    assert jws.deserialize_compact(token, key).payload == b"cupola"


def test_pillow_c_extension_loads_and_decodes() -> None:
    """pillow 13: pypdf imports PIL eagerly, so a broken wheel breaks PDF ingest.

    Cupola never calls pypdf's image paths (measured 2026-10-30: the processor
    only calls page.extract_text()), but `import pypdf` still loads
    PIL._imaging — so the extension has to work.
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(buf, format="PNG")
    buf.seek(0)
    img = Image.open(buf)
    img.load()  # forces the C decoder
    assert img.size == (4, 4)
    assert img.getpixel((0, 0)) == (1, 2, 3)


def test_every_installed_requirement_still_accepts_the_new_versions() -> None:
    """No installed distribution's pin excludes a version we just raised.

    This is the check that would have caught a real break: msal caps
    cryptography below 51 and weaviate-client caps authlib below 2.0, so the
    upgrade is only safe while those ceilings hold.
    """
    from packaging.requirements import Requirement

    raised = {k.lower(): v for k, v in PINNED_FLOORS.items()}
    conflicts = []
    checked = 0
    for dist in md.distributions():
        try:
            holder = dist.metadata["Name"]
            reqs = dist.requires or []
        except Exception:  # pragma: no cover - broken metadata in the image
            continue
        for raw in reqs:
            if "extra ==" in raw:
                continue
            try:
                req = Requirement(raw)
            except Exception:
                continue
            want = raised.get(req.name.lower())
            if want and req.specifier:
                checked += 1
                if not req.specifier.contains(Version(want), prereleases=True):
                    conflicts.append(f"{holder} requires {raw!r}, but we pin {want}")

    assert checked > 0, "no constraints were examined — the scan did not run"
    assert not conflicts, "incompatible pins:\n  " + "\n  ".join(conflicts)
