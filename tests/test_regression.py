"""run_case_ext with default arguments must reproduce the frozen notebook loop exactly."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from bridge_sim import core
from bridge_sim.ext import run_case_ext


def test_ext_matches_core():
    c = core.bridge_case(core.cfg, 19.0, 8.0)
    for kw in (dict(strategy="ilos"), dict(strategy="filtered"),
               dict(strategy="ilos", gnss_always=True)):
        a = core.run_case(c, core.truth_spec, **kw)
        b = run_case_ext(c, core.truth_spec, **kw)
        for k in ("truth", "estimate", "F", "bound", "clearance"):
            np.testing.assert_array_equal(a[k], b[k])
        assert core.outcome(a) == core.outcome(b)
        print(kw, core.outcome(b), np.nanmin(b["clearance"]))


if __name__ == "__main__":
    test_ext_matches_core()
