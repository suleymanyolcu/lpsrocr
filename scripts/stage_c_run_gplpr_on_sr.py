from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_c.pipeline import main_run_sr


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main_run_sr())

