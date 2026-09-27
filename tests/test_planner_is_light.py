"""The planner runs in a small Lambda, so it must import without OpenCV."""

import subprocess
import sys


def test_planner_imports_without_opencv():
    code = (
        "import sys; sys.modules['cv2'] = None\n"
        "from blindspot.axis import Axis\n"
        "from blindspot.boundary.planner import plan_axis\n"
        "from blindspot.boundary.criterion import Criterion\n"
        "from blindspot.cost.contract import BudgetContract\n"
        "a = Axis('exposure_ms', 'ms', 0.0, 40.0, severe_end='hi')\n"
        "print(plan_axis(a, {}, 1.25, verify_samples=5).next_values)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "40.0" in out.stdout


def test_degrade_axis_is_the_same_class():
    from blindspot.axis import Axis
    from blindspot.degrade.base import Axis as DegradeAxis

    assert Axis is DegradeAxis
