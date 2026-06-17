# Scoring and Similarity Algorithms

This document specifies the four core quantitative systems in the backend:

1. **PFV / APFV** — a single-number quality score for a player's statistical profile.
2. **NBA-to-NBA similarity** — the `similar_players` lists in `player_profiles.json`.
3. **Prospect-to-NBA similarity** — the `similar_nba_players` comps attached to draft prospects.
4. **Lineup synergy** — the synergy score, style vector, and historical-lineup matching for custom five-man units.

Sources of truth: `app/analytics/player_profiles/archetypes.py`, `app/analytics/player_profiles/similarity.py`, `app/pipelines/prospects_pipeline.py`, `app/analytics/lineup_synergy.py`, and the tuner `app/scratch/tune_points.py`.

---

## 1. PFV and APFV

### 1.1 PFV (Polygon Feature Value)

PFV measures the area of a player's six-axis radar polygon relative to the maximum possible area. The six axes, in fixed clockwise order, are:

```
pts_per36, reb_per36, ast_per36, blk_per36, stl_per36, ts_pct
```

Each axis value is the player's percentile on that metric within the reference population, rescaled to a radius

$$r_i = \frac{\text{percentile}_i}{100} \in [0, 1], \qquad i = 1, \dots, 6.$$

The polygon is drawn on $n = 6$ equally spaced spokes ($2\pi/6$ apart). Its area is the sum of the six triangles formed by adjacent radii:

$$A = \frac{1}{2}\sin\!\left(\frac{2\pi}{n}\right)\sum_{i=1}^{n} r_i\, r_{(i \bmod n)+1}.$$

The maximum area occurs when every $r_i = 1$:

$$A_{\max} = \frac{n}{2}\sin\!\left(\frac{2\pi}{n}\right).$$

$$\mathrm{PFV} = \frac{A}{A_{\max}} = \frac{\sum_{i=1}^{6} r_i\, r_{i+1}}{6} \in [0, 1].$$

Because adjacent axes multiply, PFV rewards *balanced* profiles: a player at the 80th percentile on all six axes ($\mathrm{PFV} = 0.64$) outscores a player at the 99th percentile on three alternating axes and the 10th on the others.

### 1.2 Adjusted PFV

PFV is adjusted for workload and translatability. The adjustment differs by population.

**NBA players:**

$$\mathrm{APFV}_{\text{adj}} = \mathrm{PFV} \cdot \left(\frac{\text{mpg percentile}}{100}\right)^{1.5}.$$

Per-36 production on tiny minutes is discounted superlinearly.

**Prospects** (pre-draft statistics only) use three multiplicative dampers on raw values, not percentiles:

$$\mathrm{APFV}_{\text{adj}} = \mathrm{PFV} \cdot f_{\text{mpg}} \cdot f_{\text{gp}} \cdot f_{\text{eff}}$$

with

$$f_{\text{mpg}} = \left[\min\!\left(\frac{\text{MPG}}{24},\, 1\right)\right]^{1.4}
\qquad\text{(full credit at 24+ MPG; 10 MPG} \approx 0.31\text{)},$$

$$f_{\text{gp}} = \left[\min\!\left(\frac{\text{GP}}{25},\, 1\right)\right]^{0.5} \text{ if GP} > 0 \text{, else } 1
\qquad\text{(16 GP} \approx 0.80\text{, 3 GP} \approx 0.35\text{)},$$

$$f_{\text{eff}} = \min\!\left(\frac{\mathrm{TS\%}}{0.60},\, 1\right) \text{ if TS\%} > 0 \text{, else } 1
\qquad\text{(linear ramp anchored at 0.60 true shooting)}.$$

The absolute (non-percentile) floors prevent small-sample and low-minute stat-stuffing pathologies.

### 1.3 APFV normalization

The adjusted value $v$ is normalized against a population $V = \{v_1, \dots, v_n\}$:

$$\text{rank\_score} = \left(\frac{\#\{v_j \le v\}}{n}\right)^{c}, \qquad c = 1.5 \text{ (curve exponent)}.$$

If a raw anchor $a$ is supplied, a magnitude term prevents "best of a weak bucket" from reaching the ceiling:

$$\text{raw\_score} = \mathrm{clip}\!\left(\frac{v}{a},\, 0,\, 1\right), \qquad
\mathrm{APFV} = 0.99\sqrt{\text{rank\_score} \cdot \text{raw\_score}}.$$

With no anchor, $\mathrm{APFV} = 0.99 \cdot \text{rank\_score}$. The output is capped at $0.99$.

**Height-bucket normalization.** The population is partitioned by listed height before ranking:

| Bucket | Height |
|---|---|
| guard | under 6'4" (< 76 in) |
| wing | 6'4" – 6'8" (76–80 in) |
| big | over 6'8" (> 80 in) |

Each bucket is normalized independently, so the best guard and the best big both reach approximately $0.99$. This removes the structural bias of bigs accumulating inflated rebound/block percentiles.

---

## 2. NBA-to-NBA similarity

Built in `analytics/player_profiles/similarity.py::build_similarity_index`. For each pair of NBA players $(i, j)$ the composite score is a product of soft affinity terms — any one term can veto an otherwise close match:

$$S(i, j) = A_{\text{play}} \cdot A_{\text{size}} \cdot A_{\text{caliber}} \cdot A_{\text{era}} \cdot A_{\text{role}} \cdot A_{\text{arch}}.$$

Each player's top $k = 10$ scores become their `similar_players` list, sorted descending.

### 2.1 Playstyle affinity

Twelve features in career-percentile space $x \in [0,1]^{12}$:

```
pts_per36, reb_per36, ast_per36, blk_per36, stl_per36, tov_per36,
fg3a_rate, fta_rate, ts_pct, efg_pct, ast_pct, mpg
```

with diagonal weight vector

$$w = (1.0,\, 1.9,\, 1.7,\, 1.6,\, 1.2,\, 0.5,\, 1.9,\, 1.1,\, 0.7,\, 0.6,\, 2.1,\, 0.6)$$

(high-signal stylistic axes — rebounding, three-point volume, assist share, rim protection, creation — dominate; volume and efficiency nudge). The affinity is a Laplacian kernel on weighted Euclidean distance:

$$d_{\text{play}}(i,j) = \lVert w \odot (x_i - x_j) \rVert_2, \qquad
A_{\text{play}} = \exp\!\left(-\frac{d_{\text{play}}}{\beta}\right), \quad \beta = 2.2.$$

### 2.2 Size affinity

Gaussian kernels on height $h$ (inches, $\sigma_h = 12$) and weight $m$ (lbs, $\sigma_m = 36$), combined and softened by a fourth root so playstyle dominates:

$$A_{\text{size}} = \left[
\exp\!\left(-\frac{(h_i - h_j)^2}{2\sigma_h^2}\right) \cdot
\exp\!\left(-\frac{(m_i - m_j)^2}{2\sigma_m^2}\right)
\right]^{1/4}.$$

A six-inch height mismatch costs roughly 12 percent.

### 2.3 Caliber affinity

$q \in [0,1]$ is each player's APFV (Section 1). With $\sigma_q = 0.35$:

$$A_{\text{caliber}} = \exp\!\left(-\frac{(q_i - q_j)^2}{2\sigma_q^2}\right).$$

The sigma is broad by design so role-player and star versions of the same archetype still match.

### 2.4 Cross-era reward

With $\Delta t = |{\text{first season}}_i - {\text{first season}}_j|$ in years and timescale $\tau = 14$:

$$A_{\text{era}} = 1 + 0.10\left(1 - e^{-\Delta t / \tau}\right) \in [1.0,\, 1.10].$$

This is a reward, not a penalty: same-year pairs get $1.0$ and distant eras approach $1.10$, diversifying lists that would otherwise be dominated by same-era statistical correlation.

### 2.5 Role and archetype bonuses

$$A_{\text{role}} = \begin{cases} 1.07 & \text{same assigned role} \\ 1.0 & \text{otherwise} \end{cases}
\qquad
A_{\text{arch}} = 1 + 0.18 \cdot |\,\mathcal{A}_i \cap \mathcal{A}_j\,|$$

where $\mathcal{A}_i$ is player $i$'s set of box-score archetypes. Shared archetypes ("high-volume creator", "rebounding defender", ...) cluster concept groups multiplicatively.

### 2.6 Display score

Raw composites fall roughly in $[0, 1.6]$ (the multiplicative bonuses exceed 1). The displayed percentage is

$$\text{display} = 78.0 \cdot \left[\mathrm{clip}\!\left(\frac{S}{1.45},\, 0,\, 1\right)\right]^{2.0},$$

a soft ceiling at 78 so only true-twin pairs read at the top and typical top-10 entries land in the 40–70 range.

---

## 3. Prospect-to-NBA similarity

Built in `pipelines/prospects_pipeline.py::compute_similarity_matrix` (scoring) and `_build_similarity_payload` (display selection). The scoring path is shared verbatim with the offline tuner (`scratch/tune_points.py`), so the evaluated space and the production space are identical by construction.

### 3.1 Design constraints

- **Pre-draft information only** on the prospect side: per-36 box statistics, shooting rates, minutes, listed height/weight. No draft position, no post-draft data.
- **Height/weight budget cap**: the combined squared weight of the two size dimensions may not exceed 25 percent of the total squared weight,
  $$w_h^2 + w_m^2 \le 0.25 \left( w_h^2 + w_m^2 + \textstyle\sum_f w_f^2 \right),$$
  preventing the optimizer from collapsing to a trivial size-identity match.

### 3.2 Peer-relative feature space (versioned foundation)

The central normalization idea: a prospect and an NBA player are compared by *where each stands among their own peers*, the unit that transfers across the college-to-NBA gap. **How** that standing is encoded is a versioned choice (the `feature_norm` field, §3.8). Three foundations are implemented; both sides of the comparison always use the same encoding.

Thirteen features are used (ten base, two engineered playstyle, one caliber, computed identically on both pools):

```
pts_per36, reb_per36, ast_per36, blk_per36, stl_per36,
fg3a_rate, fta_rate, ts_pct, ast_pct, mpg,
stocks        = stl_per36 + blk_per36,
scoring_load  = pts_per36 * (1 - clip(ast_pct, 0, 1)),
quality       = APFV of the player within their own pool
```

The **quality** axis is the absolute-caliber rank (Section 1, ranked across the whole pool rather than within a height bucket, so it preserves absolute rather than position-relative caliber). It is the unit that prevents an elite, balanced prospect from collapsing onto generic same-shape journeymen.

The NBA pool is players with $\ge 200$ career games; the prospect pool is the prospect's own draft class (the full board of roughly 60 players). For each feature $f$ with raw value $v$:

- **`percentile`** — empirical rank within the pool, $p = \#\{j : v_j \le v\} / |\text{pool}|$ (NBA side uses precomputed `*_global_pctile/100` where available). Robust to skew and small samples; output in $[0,1]$.
- **`zscore`** — parametric standardization within the pool, $z = (v - \mu_{\text{pool}}) / \sigma_{\text{pool}}$, using the pool mean and standard deviation. Output centered at 0 with unit spread.
- **`hybrid`** *(active, `v4_hybrid`)* — the feature vector is the **concatenation** of the percentile block and the z-score block, the latter scaled by a fixed $\zeta = 0.25$ (`HYBRID_Z_SCALE`) so the two blocks sit on a comparable spread under one shared bandwidth:
  $$x = \big[\, p_1,\dots,p_d,\; \zeta z_1,\dots,\zeta z_d \,\big] \in \mathbb{R}^{2d}.$$
  The $2d$ per-feature weights (§3.3) let the tuner decide how much each representation of each feature contributes; zeroing the z-block recovers the pure percentile engine exactly, so the hybrid subsumes it and was tuned upward from that anchor.

Height and weight are standardized against the pooled prospect-plus-NBA distribution under every foundation (they transfer in absolute terms, so they are never rank- or pool-relative):

$$\tilde h = \frac{h - \mu_h}{\sigma_h}, \qquad \tilde m = \frac{m - \mu_m}{\sigma_m}.$$

### 3.3 Weighted distance and base score

Let $x_p, x_j$ be the feature vectors of the chosen foundation (§3.2), augmented with the two standardized size dimensions, and $w$ the tuned weight vector (one weight per feature dimension, then height, weight — so length $d+2$ for `percentile`/`zscore`, $2d+2$ for `hybrid`). The base similarity is a kernel on weighted Euclidean distance:

$$d(p, j) = \lVert w \odot (x_p - x_j) \rVert_2, \qquad
s_0(p, j) = \begin{cases}
\exp\!\left(-\dfrac{d^2}{2\beta^2}\right) & \text{kernel} = \texttt{gaussian} \\[2mm]
\exp\!\left(-\dfrac{d}{\beta}\right) & \text{kernel} = \texttt{laplacian.}
\end{cases}$$

The kernel is a versioned field (§3.8). The **active `v4_hybrid`** foundation uses the Gaussian kernel with $\beta = 0.055$ and the $2d + 2 = 28$ tuned weights (percentile block, then z-score block, then height, weight):

$$\begin{aligned}
w_{\text{pct}} &= (0.0,\, 0.26,\, 0.0,\, 0.417,\, 0.184,\, 0.178,\, 0.197,\, 0.1,\, 0.347,\, 0.051,\, 0.077,\, 0.062,\, 0.122) \\
w_{\text{z}}   &= (0.1,\, 0.0,\, 0.0,\, 0.15,\, 0.1,\, 0.0,\, 0.1,\, 0.0,\, 0.0,\, 0.05,\, 0.0,\, 0.0,\, 0.0) \\
w_{\text{hw}}  &= (0.290,\, 0.319).
\end{aligned}$$

The z-block adds rim-protection, scoring, stocks, free-throw-pressure and minutes signal on top of the percentile core. The height/weight weights satisfy the budget cap (§3.1) at ratio $0.249 \le 0.25$ of the total squared weight.

### 3.4 Graph smoothing (second-order similarity)

Let $N_7(j)$ be the first seven entries of NBA player $j$'s `similar_players` list (Section 2). The final score blends each candidate's own similarity with the best similarity in its NBA-NBA neighborhood:

$$s(p, j) = (1 - \lambda)\, s_0(p, j) + \lambda \max_{k \in N_7(j)} s_0(p, k), \qquad \lambda = 0.35 \;\text{(active)}.$$

Candidates whose *neighborhoods* resemble the prospect rank higher, so the player the prospect actually becomes tends to sit at most one hop from the displayed comps. This uses NBA-side information only.

### 3.5 Tuning objective

The full foundation — `feature_norm`, `kernel`, the feature set, weights, bandwidth, and smoothing $\lambda$ — maximizes a **points** metric over the 2007–2023 draft classes ($n = 490$ prospects with an NBA counterpart in the pool), evaluated on the raw engine output *before* exact-name filtering and through the literal production scoring path (`tune_points.py` calls `compute_similarity_matrix`). Per prospect (max 9):

- **+2** if the prospect's NBA counterpart (same player, matched by normalized name) ranks in the top 7 of $s(p, \cdot)$;
- **+1** for each NBA player in that top 7 whose own top-7 NBA-NBA similars contain the counterpart (the second-order, neighborhood term that §3.4 smoothing optimizes).

The tuner maximizes total points, breaking ties by counterpart top-7 recall and then by mean counterpart rank. Optimization is random search followed by coordinate descent; the height/weight budget (§3.1, cap $0.25$) is enforced at every step, and classes 2024+ are excluded (insufficient NBA sample). For the `hybrid` foundation the weight vector has $2d$ feature dimensions and the search is seeded from the percentile solution (z-block at zero), so it provably starts at the percentile engine's score and climbs from there.

**Foundation comparison** (production code path, identical eval):

| Foundation (`feature_norm`, kernel) | Points | Counterpart top-7 | Median rank |
|---|---|---|---|
| `percentile`, laplacian (`v3_tau067`) | 1164 (26.4%) | 56.3% | 6 |
| `zscore`, gaussian (best, not shipped) | 541 (12.3%) | 26.9% | 38 |
| **`hybrid`, gaussian (`v4_hybrid`, active)** | **1208 (27.4%)** | **57.8%** | **4** |

A pure z-score foundation roughly halves counterpart recall: standardization is sensitive to the skewed, small-sample distributions of college box scores, where empirical rank is robust. The hybrid keeps the percentile core and adds a z-score block, which strictly improves all three metrics over the percentile engine. Re-running the tuner with `--norm` reproduces each row.

### 3.6 Display selection

Exactly four comps are stored per prospect. The displayed *score* is the raw composite $s(p, j)$; the display *ranking* applies multiplicative, selection-stage-only adjustments (they never touch the similarity matrix or the tuning metric):

$$R(p, j) = s(p, j) \cdot \underbrace{\left(1 - \bar{s}_j\right)}_{\text{popularity penalty}} \cdot \underbrace{\left(0.35 + 0.65\, E_j\right)}_{\text{establishment prior}} \cdot \underbrace{\exp\!\left(-\frac{\max(0,\, \sigma_p - \sigma_j)}{\tau}\right)}_{\text{scoring-volume affinity}}$$

where

- $\bar{s}_j = \frac{1}{n_p}\sum_p s(p, j)$ is candidate $j$'s mean similarity across every prospect in the run. Subtracting it multiplicatively demotes "universal attractor" players who score well against everyone; because the factor is bounded in $(0, 1]$, a dissimilar player can never outrank a similar one.
- $E_j = \frac{1}{2}\left(\mathrm{pctile}(\text{career minutes}_j) + \mathrm{pctile}(\mathrm{APFV}_j)\right)$ is a career-establishment percentile. Among similarly shaped candidates this prefers the player with the more substantial career, correcting systematic pessimism without using draft position or any prospect-side post-draft data.
- $\sigma_p, \sigma_j$ are points-per-game percentiles (prospect within its board, NBA within the comp pool). The factor is **one-sided** ($\tau = 0.20$): it fires only when a prospect outscores a candidate, demoting same-shape role-player comps for high-volume scorers (Edwards, Dybantsa) while leaving a low-volume prospect's high-scoring comps untouched, so playmakers and defenders keep their star comps. Because the box-score/efficiency features carry near-zero tuned weight (§3.3 — scoring volume does not transfer across the college→NBA gap and hurts counterpart recall), this is the layer that restores scoring face-validity without disturbing the tuned metric. Setting $\tau = 0$ disables it. The full parameter set is versioned (§3.8).

Selection then walks the ranking $R(p, \cdot)$ in descending order over the top $4 \times 10 = 40$ candidates subject to:

1. **Counterpart exclusion**: any NBA player whose normalized (diacritic- and punctuation-insensitive) name equals the prospect's is masked out, so the prospect's own NBA counterpart never appears.
2. **Era diversity**: at most 2 comps per career-era bucket (late-90s, early-2000s, late-2000s, early-2010s, late-2010s, 2020s, by career midpoint).
3. **Usage cap**: no NBA player may appear more than 2 times across all prospects in a single run (one draft class, or one current board). Earlier-processed prospects claim first; later ones fall to the next-best distinct comp. The era constraint is relaxed before the usage cap if a prospect runs short.

The four selected comps are stored and returned in decreasing order of a similarity-weighted career production index,

$$I_j = \left(1.0\,\mathrm{PPG}_j + 1.9\,\mathrm{APG}_j + 0.75\,\mathrm{RPG}_j\right) \cdot \frac{\text{display}_j}{100},$$

where each per-game rate is derived from career rates as $\mathrm{XPG}_j = \text{x\_per36}_j \cdot \text{mpg}_j / 36$ and $\text{display}_j$ is the displayed similarity percentage (Section 3.7).

### 3.7 Displayed percentage

Composites are well below 1.0 on the kernel scale, so the shared cosine-calibrated transform does not apply. Prospect comps use a concave map with the versioned exponent $\gamma$ (`similarity_gamma`, default $0.2$):

$$\text{display} = 100 \cdot \left[\mathrm{clip}(s,\, 0,\, 1)\right]^{\gamma}.$$

This is a cosmetic transform applied after ranking and selection; it never affects which comps are chosen. The displayed range recalibrates with the foundation — under the active `v4_hybrid` engine stored comps span roughly the high-30s to high-70s, widening as the Gaussian kernel separates strong from weak matches more sharply than the prior Laplacian.

### 3.8 Versioned foundation and tuning

A version is a named JSON parameter set under `app/data/tuning/versions/`, with `active.json` naming the live one. Crucially, a version captures the **algorithmic foundation**, not just coefficients:

- `feature_norm` — `percentile`, `zscore`, or `hybrid` (§3.2);
- `kernel` — `laplacian` or `gaussian` (§3.3);
- `features`, `weights`, `bandwidth`, `smooth_lambda`, `smooth_topk` (§3.3–3.4);
- the display-selection knobs of §3.6 — `similarity_gamma`, `popularity_penalty`, `max_appearances`, `establishment_floor`, `establishment_alpha`, `quality_bucket`, and the scoring-affinity `scoring_affinity_tau`.

Because the foundation fields are versioned, switching from the percentile engine to the hybrid engine — a change in the distance metric itself — is the same pointer flip as any weight tweak, not a code edit. `prospects_pipeline` overlays the active version on the code-default constants at import (`apply_tuning`/`current_tuning`); the constants are the fallback when no store is present, and `prospect_tuning.LEGACY_PARAM_DEFAULTS` supplies `feature_norm=percentile`/`kernel=laplacian` for version files written before the foundation fields existed, so older versions keep their original behavior.

Manage versions with `app/scripts/prospect_tuning_cli.py` (`list`, `show`, `diff`, `activate`, `snapshot --set key=value`, `regenerate [--version NAME]`). `regenerate` recomputes comps from the stored prospect datasets without re-scraping and rewrites the outputs, so two foundations can be compared on the same board. The tuner `scratch/tune_points.py --search --norm {percentile|zscore|hybrid} [--kernel ...]` re-derives a foundation's optimum.

Shipped lineage (all percentile/laplacian except where noted): `v1_baseline` ($\tau = 0$) → `v2_scoring_affinity` ($\tau = 0.20$) → `v3_tau067` ($\tau = 0.067$) → **`v4_hybrid`** (active; `feature_norm=hybrid`, `kernel=gaussian`, the first foundation change — see §3.5 for the gain over `v3`). Reverting to any percentile engine is `activate v3_tau067 && regenerate`.

---

## 4. Lineup synergy

Computed in `analytics/lineup_synergy.py::calculate_lineup_synergy` for an arbitrary set of five player-season rows. The output is a synergy score in $[0, 100]$, a six-axis style vector of percentiles, a factor breakdown, narrative strengths/weaknesses, and a list of similar historical starting lineups.

### 4.1 Role assignment

Each player receives up to two roles from league-relative z-scores of their season rates (thresholds applied in order; first creation role, then shot-mix role, then interior/defense):

| Role | Condition (z-scores) |
|---|---|
| Playmaker | $z_{\text{ast}} > 0.8$ and $z_{\text{ast\%}} > 0.6$ |
| Designated Scorer | else $z_{\text{pts}} > 0.8$ and $z_{\text{ast\%}} < 0.4$ |
| Secondary Creator | else $z_{\text{ast}} > 0.2$ and $z_{\text{pts}} > 0$ |
| Perimeter Specialist | $z_{\text{3pa rate}} > 0.6$ |
| Rim Attacker | else $z_{\text{fta rate}} > 0.5$ and $z_{\text{pts}} > 0$ |
| Interior Presence | ($z_{\text{reb}} > 0.7$ or $z_{\text{blk}} > 0.7$) and $z_{\text{3pa rate}} < -0.4$ |
| Defensive Specialist | ($z_{\text{stl}} > 0.8$ or $z_{\text{blk}} > 0.8$) and not Interior Presence |

A fallback assigns Secondary Creator / Interior Presence / Defensive Specialist by sign of $z_{\text{pts}}$, $z_{\text{reb}}$.

### 4.2 Collective lineup statistics

Per-36 rates are summed over the five players and scaled by $48/36 = 1.3\overline{3}$ to project a full team-game stat line: three-point attempts, paint attempts, assists, made field goals, rebounds, blocks. Derived quantities:

$$\text{ast\%}_{\text{proj}} = 100\,\frac{\mathrm{AST}_{\text{proj}}}{\mathrm{FGM}_{\text{proj}}}, \qquad
\text{paint\_score} = \mathrm{REB}_{\text{proj}} + 4\,\mathrm{BLK}_{\text{proj}}.$$

Paint attempts per player are estimated from non-three attempts with an archetype-dependent ratio:

$$\text{paint FGA} = (\mathrm{FGA} - \mathrm{3PA}) \cdot
\begin{cases}
0.80 & z_{\text{reb}} > 0.5 \text{ or } z_{\text{blk}} > 0.5 \\
0.30 & \mathrm{3PA}/\mathrm{FGA} > 0.45 \\
0.45 & \text{otherwise.}
\end{cases}$$

A per-player defense score (summed over the lineup) is

$$D = \text{stl}_{36} + 1.2\,\text{blk}_{36} +
\begin{cases} 1.5 & \text{Defensive Specialist} \\ 1.0 & \text{Interior Presence} \\ 0 & \text{otherwise.} \end{cases}$$

The same statistics are computed for every actual starting lineup of the chosen season; their means ($\overline{\mathrm{REB}}$, $\overline{D}$, $\overline{\text{paint}}$, $\overline{\mathrm{BLK}}$) serve as league baselines.

### 4.3 Style vector

Six percentiles, each ranked against the season's real teams (or real starting lineups where noted):

- **Pace**: minutes-weighted average of each player's team pace, ranked against team paces.
- **Three-point volume**: projected lineup 3PA ranked against team 3PA.
- **Paint**: estimated team paint FGA, $\;\widehat{\text{PFGA}} = \overline{\text{PFGA}}_{\text{teams}} + 15\,\frac{\text{paint\_score} - \overline{\text{paint}}}{\overline{\text{paint}}}$, clipped to $[25, 70]$, then ranked.
- **Rebounding**: estimated REB%, $\;\widehat{\mathrm{REB\%}} = \overline{\mathrm{REB\%}}_{\text{teams}} + 10\,\frac{\mathrm{REB}_{\text{proj}} - \overline{\mathrm{REB}}}{\overline{\mathrm{REB}}}$, clipped to $[40, 60]$, then ranked.
- **Defense**: estimated defensive rating $\;\widehat{\mathrm{DRtg}} = \overline{\mathrm{DRtg}}_{\text{teams}} - 15\,\frac{D - \overline{D}}{\overline{D}}$, ranked and inverted (lower rating = higher percentile).
- **Playmaking**: $0.55 \cdot \mathrm{pctile}(\mathrm{AST}_{\text{proj}} \mid \text{starting lineups}) + 0.45 \cdot \mathrm{pctile}(\text{ast\%}_{\text{proj}} \mid \text{starting lineups})$.

### 4.4 Synergy score

**Baseline talent.** Each player's rating is a weighted blend of season percentiles,

$$r_i = 0.30\,P_{\text{pts}} + 0.15\,P_{\text{ast}} + 0.15\,P_{\text{reb}} + 0.10\,P_{\text{stl}} + 0.10\,P_{\text{blk}} + 0.20\,P_{\text{ts}},$$

and the lineup baseline penalizes weak links:

$$B = 0.85\,\overline{r} + 0.15\,\min_i r_i.$$

**Additive adjustments.** With $m = \#\text{Playmakers} + 0.5 \cdot \#(\text{Secondary Creators} + \text{Designated Scorers})$, shooters counted by role or $\text{3PM}_{36} \ge 1.5$ or 3PA-rate percentile $> 65$, and defenders by role or steal/block percentile $> 80$:

| Factor | Values |
|---|---|
| Playmaking | $-15$ if $m = 0$; $0$ if $m \le 1$; $+5$ if $m \le 2.5$; $-10$ if $m > 2.5$ (crowding) |
| Spacing | $-20 / -10 / +2.5 / +6$ for $0 / 1 / 2 / 3{+}$ shooters |
| Interior | with $\rho_r = \mathrm{REB}_{\text{proj}}/\overline{\mathrm{REB}}$, $\rho_b = \mathrm{BLK}_{\text{proj}}/\overline{\mathrm{BLK}}$: $-15$ if $\rho_r < 0.88$ or $\rho_b < 0.70$; $-10$ if $\rho_r > 1.25$ and $\rho_b > 1.60$ (congestion); $+5$ if $\rho_r \ge 1.05$ and $\rho_b \ge 1.15$; else $+2$ |
| Defense | $-10 / 0 / +4$ for $0 / 1 / 2{+}$ defenders |
| Role overlap | $-6$ if any of {Designated Scorer, Interior Presence, Playmaker} appears as a primary role $\ge 3$ times |

Each role-based adjustment is then reconciled against the corresponding measured style-vector percentile (for example, a negative playmaking adjustment is softened when the measured playmaking percentile is $\ge 60$, and forced to at most $-5$ when it is below $45$; spacing, interior, and defense adjustments are clamped negative when their percentiles fall below $45$).

$$\text{synergy} = \mathrm{clip}\left(B + \Delta_{\text{play}} + \Delta_{\text{space}} + \Delta_{\text{interior}} + \Delta_{\text{def}} + \Delta_{\text{overlap}},\; 0,\; 100\right).$$

Strength/weakness narratives are emitted from the same gated conditions, capped at seven entries.

### 4.5 Similar historical lineups

Every historical starting lineup $(T, s)$ is scored against the custom unit as a blend of four affinities.

**Style affinity.** With both style vectors scaled to $[0,1]^6$ and axis weights $u = (0.75, 1.20, 1.00, 1.15, 1.20, 0.95)$ for (pace, three-point volume, paint, defense, playmaking, rebounding):

$$d = \sqrt{\frac{\sum_k u_k (a_k - b_k)^2}{\sum_k u_k}}, \qquad
A_{\text{style}} = \max\!\left(0,\; 1 - \frac{d}{0.78}\right).$$

**Role affinity.** Cosine similarity between the two seven-dimensional role-count vectors (each player contributes up to two roles).

**Quality affinity.** The synergy score maps to a target winning percentage

$$w^* = \mathrm{clip}(0.24 + 0.0062 \cdot \text{synergy},\; 0.25,\; 0.82), \qquad
A_{\text{qual}} = \max\!\left(0,\; 1 - \frac{|w_{\text{hist}} - w^*|}{0.52}\right).$$

**Player affinity.** Exact roster matches are removed first; the remaining players are matched by maximum-weight bipartite assignment (permutation search) where a pair's weight is the NBA-to-NBA `similar_players` display score divided by 100 (checked in both directions), else $0.35$ for same role, else $0.15$:

$$A_{\text{player}} = \frac{\#\text{exact} + \max_{\sigma} \sum_i \mathrm{sim}(p_i, h_{\sigma(i)})}{5}.$$

**Combination.** With exact-overlap count $e$:

$$w_p = \min\!\left(0.55,\; 0.12 + 0.10\,e + 0.22\max(0,\, A_{\text{player}} - 0.35)\right),$$

$$S_{\text{base}} = w_p A_{\text{player}} + (1 - w_p)\left(0.54\,A_{\text{style}} + 0.26\,A_{\text{role}} + 0.20\,A_{\text{qual}}\right).$$

For near-exact rosters ($A_{\text{player}} \ge 0.6$) a quadratic boost pulls the combined score toward $A_{\text{player}}$:

$$t = \left(\frac{A_{\text{player}} - 0.6}{0.4}\right)^2, \qquad
S = S_{\text{base}} + t\,(A_{\text{player}} - S_{\text{base}}),$$

and lineups sharing $e \ge 2$ actual players receive an additive bonus of $0.045\,(e - 1)$, capped at $1.0$. The displayed percentage is $100\,S$.

**Selection.** The final list of eight is chosen by a staged heuristic: first lineups with $e \ge 4$ and score $\ge 78$, then $e \ge 3$ and score $\ge 64$ (roster continuity), then the best candidate $\ge 58$ from each decade not yet represented, then greedy fill maximizing score minus diversity penalties ($-5$ per already-selected lineup from the same franchise, $-3$ per same decade, $-4$ if it shares four starters with an already-selected lineup, $+6$ if $e \ge 3$). The result is sorted by score descending.
