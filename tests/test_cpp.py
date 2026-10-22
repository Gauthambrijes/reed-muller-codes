"""Cross-checks of the C++ engine (cpp/sim) against the Python reference and theory."""

import csv
import io
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rmcodes import ReedMullerCode, RPADecoder

CPP = Path(__file__).resolve().parent.parent / "cpp"
SIM = CPP / "sim"

pytestmark = pytest.mark.skipif(shutil.which("g++") is None and not SIM.exists(),
                                reason="needs a C++ compiler or a built cpp/sim")


@pytest.fixture(scope="module", autouse=True)
def build():
    if shutil.which("make") and shutil.which("g++"):
        subprocess.run(["make", "-s", "-C", str(CPP)], check=True)
    assert SIM.exists()


def run(*args, stdin=None):
    out = subprocess.run([str(SIM), *map(str, args)], input=stdin, capture_output=True, check=True)
    return out.stdout


@pytest.mark.parametrize("r,m", [(2, 5), (2, 6), (3, 5)])
@pytest.mark.parametrize("dec", ["exact", "minsum"])
def test_cpp_rpa_matches_python(r, m, dec):
    code = ReedMullerCode(r, m)
    rng = np.random.default_rng(7)
    u = rng.integers(0, 2, (300, code.k), dtype=np.uint8)
    sigma = 0.8
    llr = 2 * (1 - 2.0 * code.encode(u) + sigma * rng.standard_normal((300, code.n))) / sigma ** 2
    raw = run("decode", "--r", r, "--m", m, "--dec", dec,
              stdin=struct.pack("<i", len(llr)) + llr.astype("<f8").tobytes())
    cpp_bits = (np.frombuffer(raw, dtype=np.int8).reshape(len(llr), code.n) < 0).astype(np.uint8)
    py_bits = RPADecoder(code, min_sum=(dec == "minsum")).decode_codeword(llr)
    same = np.all(cpp_bits == py_bits, axis=1).mean()
    assert same >= 0.99, same  # only floating-point summation order differs


@pytest.mark.parametrize("r,m,bits", [(2, 6, 4), (2, 8, 6), (3, 6, 5)])
def test_fixed_point_widths_within_proven_bounds_and_tight(r, m, bits):
    rows = list(csv.DictReader(io.StringIO(run(
        "width", "--r", r, "--m", m, "--bits", bits, "--ebn0", "0,2,4", "--blocks", 100).decode())))
    assert rows
    for row in rows:
        bound = int(row["bound"])
        adv, rnd = int(row["observed_adversarial"]), int(row["observed_random"])
        assert max(adv, rnd) <= bound, row          # never exceeds the theorem
        assert adv == bound, row                     # the bound is attained (tight)
