# Prospect Similarity — Weight Calibration

Tooling to re-tune the **prospect → NBA similarity** weights under per-column
constraints you control, measure the result against the currently-shipped
config, and keep a documented history of every weight set you've shipped.

The search runs through the real production code path
(`ProspectsPipeline.compute_similarity_matrix`), so weights tuned here behave
identically once pasted into the algorithm.

## Files

| File | Purpose |
|------|---------|
| `calibrate_weights.py` | The calibration script (baseline eval + bounded search + report + history logging). |
| `weight_bounds.json` | **You edit this.** Floors/ceilings per column, plus search settings. |
| `weights_history.json` | Append-only ledger of every shipped/tuned config and its metrics. |
| `README.md` | This file. |

## What gets optimized

The objective is the **points metric** over the 2007–2023 draft classes
(n ≈ 490 prospects with a known NBA counterpart). Per prospect (max 9):

- **+2** if the prospect's own NBA self lands in the engine's top-7 matches
  (this is the *first-order* "are the comps actually right" signal), and
- **+1** for each of those top-7 whose own NBA-NBA top-7 neighbors include the
  counterpart (*second-order* graph agreement).

Reported metrics:

- **points** / **points per prospect** — the headline objective.
- **top-7 hit %** — how often the prospect's real NBA self is in the top-7
  (the metric that most directly reflects comp quality).
- **median / mean rank** — where the true counterpart lands on average.

> ⚠️ **Known tension:** ~⅔ of total points come from the second-order graph,
> which itself clusters heavily on height/weight. So *raising* h/w tends to
> *raise* points but *hurt* first-order comp quality. The `hw_budget` knob (and
> the per-column h/w ceilings) let you hold h/w down; the report shows both
> sides so you can judge the trade.

## Setup

Requirements are already part of the app environment (`pandas`, `numpy`,
`fastparquet`). The script reads:

- `app/data/static/player_career_features.parquet` (NBA comp pool)
- `app/data/static/player_profiles.json` (NBA-NBA second-order graph)
- `app/data/static/draft/prospects_<year>.parquet` (eval classes)

These are produced by the normal pipeline; if they're stale, rebuild with
`python3 app/scripts/build_historical_prospects.py --all --recompute`.

## Workflow

### 1. See the current baseline

```bash
python3 app/calibration/calibrate_weights.py
```

Prints the shipped config's points / points-per-prospect / top-7 hit. No search.

### 2. Set your floors and ceilings

Edit **`weight_bounds.json`**. Each entry in `bounds` is `[floor, ceiling]` for
that column's weight. Columns are the 12 playstyle features plus `height` and
`weight` (the size prior).

```jsonc
"bounds": {
  "pts_per36": [0.0, 0.50],   // let pts range 0 → 0.50
  "weight":    [0.0, 0.30],   // cap the weight prior at 0.30
  "blk_per36": [0.40, 0.40]   // pin blk exactly at 0.40
}
```

Other knobs:

- `bandwidth` / `smooth_lambda`: `{ "fixed": 0.08 }` to hold constant, or
  `{ "grid": [0.08, 0.1, 0.12] }` to let the search pick.
- `hw_budget`: cap on height+weight as a share of the **squared** weight budget
  (e.g. `0.18` = 18%). Set to `null` to let the per-column bounds govern alone.
- `search`: `n_random` random samples per seed, `seeds` list, `cd_rounds`
  coordinate-descent passes, `cd_step` step size. Defaults are thorough
  (~30–60s). Lower them for a quick pass.

### 3. Run the calibration

```bash
python3 app/calibration/calibrate_weights.py --search --note "tone down ts, raise stl"
```

The search starts from the shipped weights (clamped into your box, so it can
never do worse than a bounded baseline), explores randomly, then coordinate-
descends. It prints an **improvement/downgrade** report vs the shipped config
and a paste-ready `SIMILARITY_COMP_WEIGHTS` block.

Useful flags:

- `--bounds <path>` — use a different bounds file.
- `--no-log` — don't append this run to `weights_history.json` (for experiments).
- `--note "<text>"` — label stored with the run in the ledger.

### 4. Apply the new weights

Copy the printed block into
[`app/pipelines/prospects_pipeline.py`](../pipelines/prospects_pipeline.py):

```python
SIMILARITY_COMP_WEIGHTS = np.array([ ... ])   # column order matches weight_bounds.json
SIMILARITY_COMP_BANDWIDTH = ...
SIMILARITY_COMP_SMOOTH_LAMBDA = ...
```

Column order is: the 12 `features` (in the order listed in `weight_bounds.json`)
followed by `height`, then `weight`.

### 5. Regenerate the product so the weights go live

Editing the constants only changes the algorithm — the served datasets are
pre-computed. Rebuild them:

```bash
python3 app/scripts/build_historical_prospects.py --all --recompute   # historical classes
python3 app/scripts/run_pipeline.py --stages prospect                 # current class
```

(These hit RealGM and, if configured, upload to GCS.)

## History ledger

Every `--search` run (unless `--no-log`) appends to `weights_history.json`:
timestamp, note, the full weight vector, bandwidth/lambda, `hw_budget`, the
metrics, and the delta vs the shipped config. The file is seeded with the prior
(h/w 25%) and current (h/w 18%) shipped configs so you always have a reference
point. Treat it as the source of truth for "what did we ship and how did it do."

## Notes

- Tuning is locked to **2007–2023** (`MIN_EVAL_YEAR` / `MAX_EVAL_YEAR` in the
  script) — later classes lack enough NBA track record to be reliable labels.
- Similarity reads `{feature}_global_pctile` (league-wide rank vs players with
  ≥200 career games), **not** the height-bucketed `_career_pctile` used for the
  player-profile *display*.
- The display layer (4 comps, era diversity, anti-spam, establishment prior) is
  a presentation step on top of this metric and is unaffected by these weights.
