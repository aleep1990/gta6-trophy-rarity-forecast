# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aleep1990/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aleep1990/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one: for every PlayStation 5 trophy, five quantiles of its earn rate on Exophase at the target date. A single stdlib-only Python program (`main.py`) reads a case file and writes the forecast. Rebuild v2, recalibrated on Exophase's first refresh (638 → 3,868 players tracked, 943 → 4,276 trophies earned): a completely flat history means the page is stale and a refresh is pending, so the last rate is diluted by 0.215 (the refresh totals fix the average post-refresh dilution at 0.221, and 0.215 is the robust optimum across the plausible distributions of the 103 trophies the new players earned); once the history moves, each trophy's logit rate is fitted against log-time with a robust Theil-Sen fit that uses only reads from the first refresh onward, so the stale launch-night regime cannot drag the trend and a single anomalous read cannot shift the median, and extrapolated to the target.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=8j2bwr3r&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
