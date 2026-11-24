"""Export the series behind the portfolio page's interactive charts.

    python -m scripts.export_json                         # -> ../website/assets/data/05-tda.json

Study 1: the window features on the frozen ETF prices (40-day windows, step 5) with the crisis periods.
Replication: the L1 norm of the H1 landscape and the realised volatility of 50-day windows of the four
US indices, 1992-2016, every second day, with the two crash dates of Gidea & Katz (2018).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.gidea_katz import CRASHES, END, norm_series
from scripts.make_figures import SHORT
from scripts.run_plan import build_features
from src.data import CRISIS_PERIODS, load_frozen_prices, log_returns
from src.early_warning import load_frozen_indices


def _r(values, nd: int = 5) -> list:
    return [round(float(v), nd) for v in values]


def main(out: str) -> None:
    data = build_features(load_frozen_prices())
    dates = data["returns"].index[data["end"]]
    s, topo = data["sets"], data["sets"]["topology"]
    study1 = {"dates": [str(d.date()) for d in dates],
              "volatility": _r(100 * s["volatility"][:, 0], 4),
              "eigenvalue": _r(s["eigenvalue"][:, 0], 4),
              "l1_h0": _r(topo[:, 0], 4), "l1_h1": _r(topo[:, 1], 4),
              "crises": [{"name": SHORT[k], "start": a, "end": b} for k, (a, b) in CRISIS_PERIODS.items()]}
    g = norm_series(log_returns(load_frozen_indices()).loc[:END], 50).iloc[::2]
    gk = {"dates": [str(d.date()) for d in g.index],
          "l1": _r(1e5 * g["L1"], 3), "volatility": _r(100 * g["volatility"], 4),
          "crashes": [{"name": k, "date": v} for k, v in CRASHES.items()]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"study1": study1, "gk": gk}, separators=(",", ":")))
    print(f"[ok] {out} ({Path(out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../website/assets/data/05-tda.json")
    main(ap.parse_args().out)
