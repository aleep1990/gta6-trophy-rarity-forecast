"""Forecast GTA VI trophy earn rates on Exophase at the case's target.

Entry for make.fun bounty gta6-trophy-rarity-forecast-01. Python 3.13, standard
library only, no network. Reads every <id>.in in the input directory and writes
one <id>.out per case: five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of each
trophy's earn rate at the target, scored by pinball loss on the logit scale.

The model, per trophy:
- Exophase refreshes a game's counts on its own schedule, not with every
  player's trophy. On launch night the page sat unchanged (638 players tracked,
  every rate identical) for hours, then refreshed to 3,868 players and every
  rate but the 100% one roughly 6x lower. A history that is completely flat
  (same players tracked, same rates on every read) is stale data: a refresh is
  pending, so the forecast dilutes the last rate by DILUTION and widens the
  spread to cover the refresh uncertainty.
- Once the history moves, the rates carry a real trend. It is fitted in
  logit space against log1p(hours since launch), by least squares, and read at
  the target. Log-time slows the trend naturally, so no ad-hoc damping is used.
  The spread is the fit's residual (with a floor) combined with a random-walk
  term over the horizon.

Quantiles: q_tau = inv_logit(logit(median) + z_tau * sigma), clamped to
[0.01, 99.99], rounded half-up to 4 decimals (as the kit starter rounds).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves the entry directory off sys.path; nothing beside this
# file is imported, so no path fix is needed.
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Post-refresh rate ~= pre-refresh rate * DILUTION. Launch night: 638 -> 3,868
# players tracked, 943 -> 4,276 trophies earned; new players earned ~only the
# first trophy, so the other rates fell by ~6x. 0.19 is the midpoint of pure
# dilution (0.165) and proportional spread of the extra trophies (0.221).
DILUTION = 0.19
# Logit spread while the history is flat: the refresh has not happened yet,
# so the band must be wide enough to cover where the refreshed rate lands.
SIGMA_FLAT = 0.45
# Minimum logit spread once the history moves.
SIGMA_MIN = 0.5

LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(l: float) -> float:
    return 100.0 / (1.0 + math.exp(-l))


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def _parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _linfit(xs, ys):
    """Least squares y = a + b*x. Returns (intercept, slope, residual std)."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0), 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(n - 2, 1)
    return a, b, math.sqrt(var)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    hist = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target_log = math.log1p(max(0.0, (_parse_time(case["target"]) - LAUNCH).total_seconds() / 3600.0))

    tracked = [h["players_tracked"] for h in hist]
    flat_tracked = len(set(tracked)) == 1

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # This trophy's reads, dropping nulls (a read may lack the trophy).
        pts = []
        for h in hist:
            r = h["rates"][i]
            if r is not None:
                pts.append((h["at"], r))
        if not pts:
            raise ValueError(f"{name} trophy {trophy['id']} has no rates")
        rates = [r for _, r in pts]
        r_last = rates[-1]
        flat_rates = len(set(rates)) == 1

        if flat_tracked and flat_rates:
            # Stale data: Exophase has not refreshed since launch night, and
            # a refresh is due. The 100% trophy stays earned by everyone;
            # every other rate dilutes by ~6x when the new players land.
            if r_last >= 99.99:
                median = 100.0
            else:
                median = r_last * DILUTION
            sigma = SIGMA_FLAT
        else:
            # Rich history: fit logit(rate) against log-time and extrapolate
            # to the target. Log-time slows the trend on its own.
            lx = []
            for at, _ in pts:
                hours = (_parse_time(at) - LAUNCH).total_seconds() / 3600.0
                lx.append(math.log1p(max(0.0, hours)))
            ys = [logit(r) for r in rates]
            _, slope, resid = _linfit(lx, ys)
            med_logit = logit(r_last) + slope * (target_log - lx[-1])
            median = inv_logit(med_logit)
            sigma = max(SIGMA_MIN, resid, 0.5 * math.sqrt(horizon_days))

        median = clamp(median)
        lg = logit(median)
        q = [round4(clamp(inv_logit(lg + z * sigma))) for z in Z]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
