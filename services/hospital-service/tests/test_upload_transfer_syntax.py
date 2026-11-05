"""POST /upload must hand PACS the dataset it received, under its own transfer syntax.

Two defects are pinned here.

1. The route used to assign file_meta.TransferSyntaxUID = ExplicitVRLittleEndian before the
   C-STORE. That relabels the file without re-encoding it, so compressed pixel data went to
   the archive declared as uncompressed and was stored undecodable.
2. The route used to offer every storage class in the default (uncompressed) syntaxes. The
   archive picked Implicit VR LE, so an Explicit VR LE file was converted on the way in, and a
   compressed or Big Endian file had no presentation context and was refused.

Now the association offers exactly one context: the file's SOP class in the file's own
transfer syntax. The object is stored as received or refused; it cannot be converted.

The transport is a stub (no archive is needed), so these tests pin what the route OFFERS and
SENDS. What a real archive stores is proven by tests/live_pacs_probe.py against a running
dcm4chee, which must exit 0 with the transfer syntax preserved for every fixture.
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
    offered: list = []   # (abstract syntax, [transfer syntaxes]) per add_requested_context call

    def __init__(self, *args, **kwargs):
        pass

    def add_requested_context(self, abstract_syntax, transfer_syntax=None):
        syntaxes = transfer_syntax if isinstance(transfer_syntax, (list, tuple)) else [transfer_syntax]
        _AE.offered.append((str(abstract_syntax), [None if ts is None else str(ts) for ts in syntaxes]))

    def associate(self, *args, **kwargs):
        return _Assoc(_AE.sent)


@pytest.fixture()
def client(main_module, monkeypatch):
    from fastapi.testclient import TestClient

    _AE.sent, _AE.offered = [], []
    monkeypatch.setattr(main_module, "AE", _AE)
    return TestClient(main_module.app)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_upload_offers_only_the_files_own_sop_class_and_transfer_syntax(client, name):
    blob, _ = build(name)
    original = pydicom.dcmread(io.BytesIO(blob))

    client.post("/upload", files={"file": (f"{name}.dcm", blob, "application/dicom")})

    # One context, one syntax: nothing the archive could choose instead of the original.
    assert _AE.offered == [(str(original.SOPClassUID), [str(original.file_meta.TransferSyntaxUID)])]


def test_upload_refuses_a_file_with_no_transfer_syntax_instead_of_guessing(client):
    blob, _ = build("monochrome1-explicit-le")
    bare = pydicom.dcmread(io.BytesIO(blob))
    del bare.file_meta.TransferSyntaxUID
    buffer = io.BytesIO()
    pydicom.filewriter.dcmwrite(buffer, bare, implicit_vr=False, little_endian=True,
                                enforce_file_format=False)

    response = client.post("/upload", files={"file": ("bare.dcm", buffer.getvalue(), "application/dicom")})

    assert "error" in response.json()
    assert _AE.offered == [] and _AE.sent == []


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


def test_refused_association_is_reported_and_names_the_transfer_syntax(client, main_module, monkeypatch):
    class _RefusingAE(_AE):
        def associate(self, *args, **kwargs):
            return types.SimpleNamespace(is_established=False)

    monkeypatch.setattr(main_module, "AE", _RefusingAE)
    blob, _ = build("deflated-explicit-le")

    answer = client.post("/upload", files={"file": ("deflated.dcm", blob, "application/dicom")}).json()

    assert "message" not in answer
    assert "1.2.840.10008.1.2.1.99" in answer["error"]
    assert "never converted" in answer["error"]
    assert _AE.sent == []
