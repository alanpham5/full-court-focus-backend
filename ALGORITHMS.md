# Scoring and Similarity Algorithms

This document specifies the four core quantitative systems in the backend:

1. **PFV / APFV** — a single-number quality score for a player's statistical profile.
2. **NBA-to-NBA similarity** — the `similar_players` lists in `player_profiles.json`.
3. **Prospect-to-NBA similarity** — the `similar_nba_players` comps attached to draft prospects.
4. **Lineup synergy** — the synergy score, style vector, and historical-lineup matching for custom five-man units.

Sources of truth: `app/analytics/player_profiles/archetypes.py`, `app/analytics/player_profiles/similarity.py`, `app/pipelines/prospects_pipeline.py`, `app/analytics/lineup_synergy.py`, and the tuner `app/scratch/tune_points.py`.

---

## 1. PFV and APFV

PFV and APFV are a player's single-number quality scores. For established NBA players they measure **impact** and **versatility** (§1.1–1.2), engineered so the leaderboard correlates with plus-minus impact estimators (LEBRON, RAPTOR, RAPM). A legacy *polygon* PFV (§1.4) is retained for the prospect display and the two similarity caliber axes, which are tuned against it.

### 1.1 PFV — impact and versatility

Built in `archetypes.py::calculate_impact_pfv`. The foundation is the **global** (cross-position) percentile of each playstyle axis, rescaled to $p \in [0,1]$. Global rather than position-relative percentiles preserve absolute caliber — a center is measured against the whole league, not only other centers.

Each percentile is first shrunk toward the median by a sample-size credibility $c = \min\!\left(1,\sqrt{\text{career minutes}/2500}\right)$ (so roughly one full starter season earns full credibility, and an impactful rookie's rates are largely trusted while fractional-season samples regress):

$$\tilde p = 0.5 + c\,(p - 0.5),$$

so gaudy per-36 lines on tiny samples regress to average.

From the shrunk percentiles, five intermediate skills are derived (efficiency-gating ensures volume only counts when it is efficient; the defense axis is an OR-gate so a player can be elite through steals *or* blocks):

$$\begin{aligned}
\text{scoring} &= p_{\text{pts}}\,(0.55 + 0.45\,p_{\text{ts}}) & \text{(volume gated by efficiency)}\\
\text{playmaking} &= 0.6\,p_{\text{ast}} + 0.4\,p_{\text{ast\%}}\\
\text{spacing} &= p_{\text{3pa}}\,(0.4 + 0.6\,p_{\text{efg}}) & \text{(volume gated by accuracy)}\\
\text{ball security} &= 1 - p_{\text{tov}}\\
\text{defense} &= 1 - (1 - p_{\text{stl}})(1 - p_{\text{blk}}) & \text{(steals OR blocks)}
\end{aligned}$$

**Impact** is a box-weighted sum whose loadings mirror box plus-minus structure — scoring and playmaking lead, and steals outweigh blocks (steals correlate more strongly with RAPM):

$$I = 0.32\,\text{scoring} + 0.18\,\text{playmaking} + 0.08\,p_{\text{ts}} + 0.09\,p_{\text{reb}} + 0.12\,p_{\text{stl}} + 0.07\,p_{\text{blk}} + 0.08\,\text{spacing} + 0.06\,\text{ball security}.$$

**Versatility** is the geometric mean of five macro-skills — a *no-weakness* breadth measure (one weak axis drags the whole product down), where

$$\text{creation} = 0.5\max(\text{scoring}, \text{playmaking}) + 0.5\,(0.6\,\text{scoring} + 0.4\,\text{playmaking}),$$

$$V = \left(\text{creation}\cdot p_{\text{ts}}\cdot \text{defense}\cdot p_{\text{reb}}\cdot \text{spacing}\right)^{1/5}.$$

PFV blends the two, impact-dominant:

$$\mathrm{PFV} = I^{0.72}\,V^{0.28} \in [0, 1].$$

A balanced two-way star (Jokić, LeBron) scores high on both terms; a one-dimensional specialist scores on impact but is held back on versatility; a low-usage efficient role player scores on neither. The OR-gate defense and global percentiles together stop balanced bigs from out-scoring more impactful perimeter creators.

### 1.2 APFV — workload and longevity

Built in `archetypes.py::calculate_impact_apfv_batch`. PFV is a rate-quality score; APFV layers on workload and longevity, then ranks across the **entire** pool. The adjusted value scales PFV by a factor that blends per-game role with a *saturating* career-volume term $\ell$ (`archetypes.py::longevity_factor`):

$$\text{adj} = \mathrm{PFV}\cdot\left(\frac{\text{mpg percentile}}{100}\right)^{0.50}\cdot \ell^{0.25}, \qquad
\ell = \min\!\left(1,\; \sqrt{\frac{\text{career minutes}}{4000}}\right).$$

The longevity term is a **light small-sample gate, not a compiler's reward**: $\ell$ saturates at $1$ after roughly a full starter season (4000 minutes) and enters with a small exponent ($0.25$), so it only damps fractional-season samples. Rate quality (PFV, now trusted after one full season via the §1.1 credibility) and per-game role (mpg) drive the ordering; a single impactful rookie season is therefore enough to earn a high APFV, while a top-PFV young star is no longer buried beneath a long-career compiler. (An earlier version ranked on raw career-minutes *percentile* raised to $0.60$, which scaled linearly with accumulation and pushed impactful rookies far down the board.) APFV is the global rank of $\text{adj}$ with a gentle top-end curve, capped at $0.99$:

$$\mathrm{APFV} = 0.99\left(\frac{\#\{j: \text{adj}_j \le \text{adj}\}}{n}\right)^{1.2}.$$

Ranking over a single pool (not by height bucket) preserves absolute, cross-position impact ordering, so the leaderboard reads like an impact-metric ranking rather than a best-per-position one. As with any box-derived metric, players whose value is mostly non-box (e.g., pure rim-protecting or screen-and-roll-gravity bigs) sit somewhat below their RAPM reputation — the known box-vs-RAPM gap.

**Per-season APFV** uses exactly this computation with the pool set to a single season's players (within-season percentiles, season credibility) — it is what the profile-page season selector shows (`season_profiles.py::build_season_bundle`).

**Career APFV** (the number on the default career profile) is **not** a separate ranking of career-aggregate rates. It is the **minutes-weighted mean of the player's per-season APFVs** (`season_profiles.py::career_apfv_from_seasons`):

$$\mathrm{APFV}_{\text{career}} = \frac{\sum_s \mathrm{APFV}_s \cdot \mathrm{MIN}_s}{\sum_s \mathrm{MIN}_s}.$$

Because it is a weighted average of the season values, a player's career APFV always lies within the range of their actual seasons — **it can never exceed their best season** — and sits on the same scale as the season selector. Ranking career-aggregate rates against the all-career pool (the previous approach) inflated efficient, low-usage role players: their smoothed career per-36 line ranked elite globally even though no single season did, producing a career APFV above every individual season and disagreeing with both the season values and the position-relative radar (§1.3). The minutes weighting also keeps the metric impact- rather than longevity-centric — a player with one dominant season scores near that season, while many mediocre seasons average out low. Players with no season-feature rows fall back to the career-aggregate ranking above.

**Prospects.** Pre-draft players use the same impact + versatility PFV (§1.1), with percentiles taken within the combined current-plus-historical prospect population (`ast_pct` and `tov` are unavailable pre-draft and default to neutral). Longevity does not apply, so in place of the career-volume term the adjusted value applies three multiplicative translatability dampers on *raw* values — minutes, games played, and efficiency (`archetypes.py::calculate_prospect_impact_adjusted_pfv`):

$$\text{adj} = \mathrm{PFV}\cdot f_{\text{mpg}}\cdot f_{\text{gp}}\cdot f_{\text{eff}}, \quad
f_{\text{mpg}} = \left[\min\!\left(\tfrac{\text{MPG}}{24}, 1\right)\right]^{1.4},\;
f_{\text{gp}} = \left[\min\!\left(\tfrac{\text{GP}}{25}, 1\right)\right]^{0.5},\;
f_{\text{eff}} = \min\!\left(\tfrac{\mathrm{TS\%}}{0.60}, 1\right)$$

($f_{\text{gp}}, f_{\text{eff}}$ default to 1 when GP or TS% is 0). Prospect APFV is then a height-bucketed rank with raw anchor (the normalization described in §1.4, curve exponent 2.2, anchor 0.50) across the combined prospect population. The prospect **similarity** quality axis (§3.2) still uses the legacy polygon metric, so swapping the display PFV/APFV to the impact model leaves prospect comps unchanged.

### 1.3 Displayed playstyle metrics

`playstyle_metrics` (via `style_summary`) carries, per axis, the raw per-36 value and a **global** (cross-position), credibility-shrunk percentile — the same basis PFV/APFV is built on (§1.1). A skill a player is *not* known for therefore reads low in absolute terms: a small guard's blocks/rebounds sit in the 30s-40s, not the 80s-90s they reached when ranked only against other guards (height-bucketing simultaneously inflated their weak axes and suppressed their real ones — e.g. assists). `tov_per36` is inverted so a high percentile always reads as "good." Percentiles are shrunk toward the median by the player's sample-size credibility ($\min(1,\sqrt{\text{career minutes}/5000})$), so this is the position-agnostic, sample-adjusted view APFV scores on — keeping the per-stat tiles and the radar consistent with APFV. Season-scoped profiles (`season_profiles.py::build_season_bundle`) carry the same, ranked within the single season (season credibility threshold 1500 minutes).

**Fingerprint radar (skill ratings).** The six spokes of the radar are *not* plotted as these raw percentiles. An equal-area hexagon of raw percentiles cannot track an impact-weighted APFV (a pure scorer has two strong axes and an otherwise empty radar yet a high APFV), which made radar area and APFV visibly disagree. Instead the client (`PlayerProfile.jsx::radarData`) plots a **skill rating** that recenters these global percentiles on the player's APFV while preserving shape. With raw axis percentiles $p_i$, their mean $\bar p$, the player's $\mathrm{APFV}\in[0,100]$, and a shape-contrast $k = 0.75$:

$$\text{rating}_i = \mathrm{clip}\!\left(\mathrm{APFV} + k\,(p_i - \bar p),\; 0,\; 100\right).$$

The six ratings average to APFV (exactly, absent clipping), so **two players with the same APFV cover the same area**; the *shape* still reflects relative strengths (which skills stand out among like-sized peers). The same transform runs for career and season radars (recentering on the respective APFV). Because the displayed value is no longer a percentile, the tooltip labels it a "skill rating," not a percentile.

### 1.4 Legacy polygon PFV (caliber axes)

Retained in `archetypes.py::calculate_pfv` / `calculate_adjusted_pfv` / `calculate_apfv` / `calculate_apfv_batch[_by_height]`, and used only by the two similarity caliber axes — the prospect-to-NBA **quality** axis (§3.2) and the NBA-to-NBA **caliber** axis (§2.3) — which are tuned against this metric and intentionally left unchanged. (Both the NBA-player and prospect PFV/APFV *displays* now use the impact + versatility model of §1.1–1.2; only these similarity axes remain on the polygon. The `apfv` rank normalization below is also shared by the prospect display.)

**Polygon PFV** measures the area of a player's six-axis radar polygon ($pts, reb, ast, blk, stl, ts$ as percentile radii $r_i$) relative to its maximum:

$$\mathrm{PFV}_{\text{poly}} = \frac{\sum_{i=1}^{6} r_i\, r_{(i \bmod 6)+1}}{6} \in [0, 1].$$

Because adjacent axes multiply, it rewards *balance* (the property the §1.1 model deliberately reweights toward impact for the headline score).

**Adjusted polygon PFV** is workload-adjusted, by population. NBA players: $\mathrm{PFV}_{\text{poly}}\cdot(\text{mpg percentile}/100)^{1.5}$. Prospects (pre-draft statistics only) use three multiplicative dampers on raw values:

$$\mathrm{PFV}_{\text{poly}} \cdot f_{\text{mpg}} \cdot f_{\text{gp}} \cdot f_{\text{eff}}, \quad
f_{\text{mpg}} = \left[\min\!\left(\tfrac{\text{MPG}}{24}, 1\right)\right]^{1.4},\;
f_{\text{gp}} = \left[\min\!\left(\tfrac{\text{GP}}{25}, 1\right)\right]^{0.5},\;
f_{\text{eff}} = \min\!\left(\tfrac{\mathrm{TS\%}}{0.60}, 1\right)$$

($f_{\text{gp}}, f_{\text{eff}}$ default to 1 when GP or TS% is 0). The absolute floors prevent small-sample stat-stuffing.

**Normalization.** The adjusted value $v$ is ranked against a population: $\text{rank\_score} = (\#\{v_j \le v\}/n)^{c}$, $c = 1.5$. With a raw anchor $a$, $\mathrm{APFV} = 0.99\sqrt{\text{rank\_score}\cdot\mathrm{clip}(v/a, 0, 1)}$; otherwise $\mathrm{APFV} = 0.99\cdot\text{rank\_score}$, capped at $0.99$. Ranking is either global (`calculate_apfv_batch`) or partitioned by height bucket (`calculate_apfv_batch_by_height`):

| Bucket | Height |
|---|---|
| guard | under 6'4" (< 76 in) |
| wing | 6'4" – 6'8" (76–80 in) |
| big | over 6'8" to under 7'0" (81–83 in) |
| center | 7'0" and over (≥ 84 in) |

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

$q \in [0,1]$ is each player's caliber, computed by `_caliber_array` as the **legacy** adjusted polygon PFV (§1.4) over seven core axes ($pts, reb, ast, blk, stl, ts, mpg$ in global-percentile space), normalized to a height-bucketed rank. This is a workload-aware caliber for matching, and is deliberately distinct from the headline impact APFV of §1.2 — keeping the tuned similarity behavior stable. With $\sigma_q = 0.35$:

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

The **quality** axis is the absolute-caliber rank — the legacy adjusted polygon PFV/APFV of §1.4, ranked across the whole pool rather than within a height bucket (`SIMILARITY_COMP_QUALITY_BUCKET = False`), so it preserves absolute rather than position-relative caliber. (It uses the legacy metric, not the impact APFV of §1.2, so the tuned comp behavior is unchanged.) It is the unit that prevents an elite, balanced prospect from collapsing onto generic same-shape journeymen.

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
\begin{cases} 1.5 & \text{primary role Defensive Specialist} \\ 1.0 & \text{primary role Interior Presence} \\ 0 & \text{otherwise.} \end{cases}$$

The same statistics are computed for every actual starting lineup of the chosen season; their means ($\overline{\mathrm{REB}}$, $\overline{D}$, $\overline{\text{paint}}$, $\overline{\mathrm{BLK}}$) serve as league baselines.

### 4.3 Style vector

Six percentiles, each ranked against the season's real teams (or real starting lineups where noted):

- **Pace**: minutes-weighted average of each player's team pace, ranked against team paces.
- **Three-point shooting**: lineup spacing measured by *ability* rather than volume. Each player's three-point gravity is a credibility-shrunk 3PT%, $\;g_i = (\mathrm{3PM}_i + k\,\ell)/(\mathrm{3PA}_i + k)$ with prior strength $k = 40$ and league mean $\ell = 0.355$; players below a minimum attempt rate ($\mathrm{3PA}_{36} < 1.5$) are treated as non-shooters and floored to $g = 0.30$ so a low-volume hot streak is not mistaken for floor spacing. The lineup spacing value is $\overline{g}$ across the five players, ranked against comparable starting units.
- **Paint**: estimated team paint FGA, $\;\widehat{\text{PFGA}} = \overline{\text{PFGA}}_{\text{teams}} + 15\,\frac{\text{paint\_score} - \overline{\text{paint}}}{\overline{\text{paint}}}$, clipped to $[25, 70]$, then ranked.
- **Rebounding**: estimated REB%, $\;\widehat{\mathrm{REB\%}} = \overline{\mathrm{REB\%}}_{\text{teams}} + 10\,\frac{\mathrm{REB}_{\text{proj}} - \overline{\mathrm{REB}}}{\overline{\mathrm{REB}}}$, clipped to $[40, 60]$, then ranked.
- **Defense**: estimated defensive rating $\;\widehat{\mathrm{DRtg}} = \overline{\mathrm{DRtg}}_{\text{teams}} - 15\,\frac{D - \overline{D}}{\overline{D}}$, ranked and inverted (lower rating = higher percentile). Because this box-score measure (steals + blocks) understates point-of-attack and scheme defense, the displayed percentile is lifted when it falls below $45$ but the roster carries multiple identified defenders (see §4.4): to at least $68$ for $\ge 3$ defenders and at least $62$ for $\ge 2$. The displayed style axis and the strength/weakness narratives use this reconciled percentile.
- **Playmaking**: $0.55 \cdot \mathrm{pctile}(\mathrm{AST}_{\text{proj}} \mid \text{starting lineups}) + 0.45 \cdot \mathrm{pctile}(\text{ast\%}_{\text{proj}} \mid \text{starting lineups})$.

### 4.4 Synergy score

**Baseline talent.** Each player's rating is a weighted blend of season percentiles,

$$r_i = 0.30\,P_{\text{pts}} + 0.15\,P_{\text{ast}} + 0.15\,P_{\text{reb}} + 0.10\,P_{\text{stl}} + 0.10\,P_{\text{blk}} + 0.20\,P_{\text{ts}},$$

and the lineup baseline penalizes weak links:

$$B = 0.88\,\overline{r} + 0.12\,\min_i r_i.$$

**Additive adjustments.** With $m = \#\text{Playmakers} + 0.5 \cdot \#(\text{Secondary Creators} + \text{Designated Scorers})$, shooters counted as players whose three-point gravity $g_i \ge 0.345$ (a credible league-average-or-better shooter on real volume; see §4.3), and defenders by primary role Defensive Specialist or steal percentile $> 60$ or block percentile $> 65$:

| Factor | Values |
|---|---|
| Playmaking | with $n_p = \#\text{Playmakers}$: $-15$ if $m = 0$; $-1$ if $n_p \ge 3$ (mild crowding, true playmakers only); $+2$ if $m \le 1$; else $+7$ |
| Spacing | $-20 / -10 / +4 / +8$ for $0 / 1 / 2 / 3{+}$ shooters |
| Interior | with $\rho_r = \mathrm{REB}_{\text{proj}}/\overline{\mathrm{REB}}$, $\rho_b = \mathrm{BLK}_{\text{proj}}/\overline{\mathrm{BLK}}$: $-15$ if $\rho_r < 0.88$ or $\rho_b < 0.70$; $-10$ if $\rho_r > 1.25$ and $\rho_b > 1.60$ (congestion); $+5$ if $\rho_r \ge 1.05$ and $\rho_b \ge 1.15$; else $+2$ |
| Defense | $-8 / +2 / +6$ for $0 / 1 / 2{+}$ defenders |
| Role overlap | $-6$ if any of {Designated Scorer, Interior Presence, Playmaker} appears as a primary role $\ge 3$ times |

Each role-based adjustment is then reconciled against the corresponding measured style-vector percentile (for example, a negative playmaking adjustment is softened when the measured playmaking percentile is $\ge 60$, and forced to at most $-3$ when it is below $45$; spacing and interior adjustments are clamped to at most $-3$ when their percentiles fall below $45$). The defense axis is reconciled asymmetrically: when its percentile is below $45$ — where the box-score steal/block measure structurally understates point-of-attack and scheme defense — a lineup carrying $\ge 3$ identified defenders is lifted to at least $+4$ (and its displayed percentile to $\ge 68$) and one with $\ge 2$ to at least $+1$ (percentile $\ge 62$), and only lineups with fewer identified defenders are floored to $-3$. Narratives key off these reconciled percentiles, so the displayed style vector, the factor adjustments, and the strengths/weaknesses stay mutually consistent.

$$\text{synergy} = \mathrm{clip}\left(B + \Delta_{\text{play}} + \Delta_{\text{space}} + \Delta_{\text{interior}} + \Delta_{\text{def}} + \Delta_{\text{overlap}},\; 0,\; 100\right).$$

Strength/weakness narratives are emitted from the same gated conditions, capped at seven entries.

### 4.5 Similar historical lineups

Every historical starting lineup $(T, s)$ is scored against the custom unit as a blend of four affinities.

**Style affinity.** With both style vectors scaled to $[0,1]^6$ and axis weights $u = (0.75, 1.20, 1.00, 1.15, 1.20, 0.95)$ for (pace, three-point shooting, paint, defense, playmaking, rebounding):

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
