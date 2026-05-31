"""Regression test for wheel packaging (issue #39).

Ensures all subpackages are included in the built wheel.
The original bug: setup.py hardcoded packages=['tocount'], omitting
subpackages from the PyPI wheel and causing ModuleNotFoundError.
"""
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest


EXPECTED_PACKAGES = [
    "tocount",
    "tocount.rule_based",
    "tocount.qwen_qwq",
    "tocount.deepseek_r1",
    "tocount.llama_3_1",
    "tocount.tiktoken_cl100k",
    "tocount.tiktoken_o200k",
    "tocount.tiktoken_r50k",
]


def test_wheel_contains_all_subpackages():
    """Build wheel and verify every expected subpackage is present."""
    project_root = Path(__file__).parent.parent
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", str(project_root),
             "--no-deps", "-w", tmpdir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"Could not build wheel:\n{result.stderr}")

        wheel = next(Path(tmpdir).glob("*.whl"))
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            for pkg in EXPECTED_PACKAGES:
                pkg_dir = pkg.replace(".", "/") + "/"
                assert any(
                    name.startswith(pkg_dir) for name in names
                ), f"{pkg} missing from wheel"
