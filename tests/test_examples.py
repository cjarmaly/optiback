"""Smoke tests for example scripts."""

import subprocess
import sys
from pathlib import Path


def test_delta_hedge_walkthrough_synthetic():
    script = Path(__file__).parent.parent / "examples" / "delta_hedge_walkthrough.py"
    output = Path(__file__).parent / "_equity_smoke.png"

    try:
        result = subprocess.run(
            [sys.executable, str(script), "--synthetic", "--output", str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert output.exists()
        assert "Sharpe" in result.stdout or "sharpe_ratio" in result.stdout
    finally:
        if output.exists():
            output.unlink()
