"""Every module in ai_inference must at least parse.

Pinned because a T3b edit left ``ai_worker.py`` with an unterminated string literal, and no
test imported that module: the image would have built and the worker would have died on
start. Parsing is not importing (several modules need torch or Flask), so this runs anywhere.
"""
import ast
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = sorted(
    p for p in ROOT.rglob("*.py")
    if "tests" not in p.parts and "__pycache__" not in p.parts
)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: os.path.relpath(p, ROOT))
def test_module_parses(path):
    ast.parse(path.read_text(encoding="utf-8"), str(path))


def test_the_sweep_saw_the_modules_it_exists_for():
    names = {os.path.relpath(p, ROOT).replace(os.sep, "/") for p in MODULES}
    for required in ("server.py", "ai_worker.py", "inference_engine.py", "dicom/sr_builder.py",
                     "models/modality_adapters.py", "models/study_description.py"):
        assert required in names, required
