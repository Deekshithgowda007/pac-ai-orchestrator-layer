"""POST /upload must hand PACS the dataset it received: same TransferSyntaxUID, same
PixelData bytes, same decoded pixels.

The route used to assign file_meta.TransferSyntaxUID = ExplicitVRLittleEndian before the
C-STORE. That relabels the file without re-encoding it, so compressed pixel data went to the
archive declared as uncompressed, and Big Endian pixels as Little Endian.

The transport is a stub (no archive is needed), so these tests pin what the route SENDS.
What a real archive then does is shown by tests/live_pacs_probe.py. Run against dcm4chee-arc
5.34.0 on 2026-11-05 it showed, for this route after the fix: compressed and Big Endian files
are REFUSED at association ("No presentation context ... accepted") and nothing is stored,
where before the fix they were stored undecodable; uncompressed files are stored with
identical pixels but re-encoded by the C-STORE negotiation as Implicit VR Little Endian.
Storing a file under its own transfer syntax needs a presentation context built from the
file itself, which this change does not add.
"""
import io
import pathlib
import re
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pydicom
import pytest

from dicom_fixtures import FIXTURES, build, pixel_sha256


@pytest.fixture(scope="module")
def main_module():
    # main.py loops on psycopg2.connect at import; give it a connection that answers.
    real = sys.modules.get("psycopg2")
    fake = types.ModuleType("psycopg2")
    fake.connect = lambda **_: MagicMock()
    extras = types.ModuleType("psycopg2.extras")
    extras.Json = lambda value: value
    fake.extras = extras
    sys.modules["psycopg2"], sys.modules["psycopg2.extras"] = fake, extras
    sys.modules.pop("main", None)
    import main
    yield main
    sys.modules.pop("main", None)
    sys.modules.pop("psycopg2.extras", None)
    sys.modules.pop("psycopg2", None)
    if real is not None:
        sys.modules["psycopg2"] = real


class _Assoc:
    is_established = True

    def __init__(self, sent):
        self._sent = sent

    def send_c_store(self, dataset):
        self._sent.append(dataset)
        return "0x0000"

    def release(self):
        pass


class _AE:
    sent: list = []

    def __init__(self, *args, **kwargs):
        pass

    def add_requested_context(self, *args, **kwargs):
        pass

    def associate(self, *args, **kwargs):
        return _Assoc(_AE.sent)


@pytest.fixture()
def client(main_module, monkeypatch):
    from fastapi.testclient import TestClient

    _AE.sent = []
    monkeypatch.setattr(main_module, "AE", _AE)
    return TestClient(main_module.app)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_upload_sends_the_dataset_unaltered(client, name):
    blob, pixels = build(name)
    original = pydicom.dcmread(io.BytesIO(blob))

    response = client.post("/upload", files={"file": (f"{name}.dcm", blob, "application/dicom")})

    assert response.status_code == 200
    assert "error" not in response.json(), response.json()
    assert len(_AE.sent) == 1, "the route must reach C-STORE exactly once"
    sent = _AE.sent[0]
    assert str(sent.file_meta.TransferSyntaxUID) == str(original.file_meta.TransferSyntaxUID)
    assert sent.PhotometricInterpretation == original.PhotometricInterpretation
    assert bytes(sent.PixelData) == bytes(original.PixelData)
    assert np.array_equal(sent.pixel_array, pixels)
    assert pixel_sha256(sent) == pixel_sha256(original)


def test_route_source_never_assigns_transfer_syntax():
    """Belt to the behavioural test: the assignment must not come back in any form."""
    source = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert not re.search(r"TransferSyntaxUID\s*=[^=]", source)
