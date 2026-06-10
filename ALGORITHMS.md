# Scoring and Similarity Algorithms

This document specifies the three core quantitative systems in the backend:

1. **PFV / APFV** — a single-number quality score for a player's statistical profile.
2. **NBA-to-NBA similarity** — the `similar_players` lists in `player_profiles.json`.
3. **Prospect-to-NBA similarity** — the `similar_nba_players` comps attached to draft prospects.

Sources of truth: `app/analytics/player_profiles/archetypes.py`, `app/analytics/player_profiles/similarity.py`, `app/pipelines/prospects_pipeline.py`, and the tuner `app/scratch/tune_points.py`.

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

### 3.2 Peer-percentile feature space

The central normalization idea: a prospect and an NBA player are compared by *where each ranks among their own peers*, the unit that transfers across the college-to-NBA gap.

- **NBA side**: percentile of the career value within the NBA comp pool (players with $\ge 200$ career games), using precomputed `*_career_pctile` columns where available, divided by 100.
- **Prospect side**: percentile within the prospect's own draft class (the full board of roughly 60 players),
  $$p_{if} = \frac{\#\{j \in \text{class} : v_{jf} \le v_{if}\}}{|\text{class}|}.$$

Both sides therefore live in $[0,1]^d$ and answer the same question.

Twelve playstyle features are used (ten base, two engineered, computed identically on both pools):

```
pts_per36, reb_per36, ast_per36, blk_per36, stl_per36,
fg3a_rate, fta_rate, ts_pct, ast_pct, mpg,
stocks        = stl_per36 + blk_per36,
scoring_load  = pts_per36 * (1 - clip(ast_pct, 0, 1))
```

Height and weight are standardized against the pooled prospect-plus-NBA distribution:

$$\tilde h = \frac{h - \mu_h}{\sigma_h}, \qquad \tilde m = \frac{m - \mu_m}{\sigma_m}.$$

### 3.3 Weighted distance and base score

With the tuned weight vector (order matching the feature list, then height, weight)

$$w = (0.0,\, 0.34,\, 0.05,\, 0.4,\, 0.3,\, 0.29,\, 0.31,\, 0.0,\, 0.35,\, 0.1,\, 0.13,\, 0.15,\, 0.294,\, 0.392)$$

the base similarity between prospect $p$ and NBA player $j$ is a Laplacian kernel on weighted Euclidean distance:

$$d(p, j) = \lVert w \odot (z_p - z_j) \rVert_2, \qquad
s_0(p, j) = \exp\!\left(-\frac{d(p, j)}{\beta}\right), \quad \beta = 0.08.$$

The height/weight weights satisfy the 25 percent budget cap: $0.294^2 + 0.392^2 = 0.240$ against a playstyle sum of squares of $0.719$.

### 3.4 Graph smoothing (second-order similarity)

Let $N_7(j)$ be the first seven entries of NBA player $j$'s `similar_players` list (Section 2). The final score blends each candidate's own similarity with the best similarity in its NBA-NBA neighborhood:

$$s(p, j) = (1 - \lambda)\, s_0(p, j) + \lambda \max_{k \in N_7(j)} s_0(p, k), \qquad \lambda = 0.8.$$

Candidates whose *neighborhoods* resemble the prospect rank higher, so the player the prospect actually becomes tends to sit at most one hop from the displayed comps. This uses NBA-side information only.

### 3.5 Tuning objective

Parameters (feature set, weights, bandwidth, smoothing) maximize a points metric over the 2007–2023 draft classes ($n = 490$ prospects with an NBA counterpart in the pool), evaluated on the raw engine output *before* exact-name filtering:

- **+2** if the prospect's NBA counterpart (same player, matched by normalized name) ranks in the top 7 of $s(p, \cdot)$;
- **+1** for each NBA player in that top 7 whose own top-7 NBA-NBA similars contain the counterpart.

Maximum 9 points per prospect. Optimization is random search followed by coordinate descent, with the height/weight budget constraint enforced at every step. Classes 2024 and later are excluded (insufficient NBA sample). The shipped configuration scores 1150 of 4410 (26.1 percent), with the counterpart in the raw top 7 for 40.8 percent of prospects.

### 3.6 Display selection

Exactly four comps are stored per prospect. The displayed *score* is the raw composite $s(p, j)$; the display *ranking* applies three multiplicative, selection-stage-only adjustments (they never touch the similarity matrix or the tuning metric):

$$R(p, j) = s(p, j) \cdot \underbrace{\left(1 - \bar{s}_j\right)}_{\text{popularity penalty}} \cdot \underbrace{\left(0.35 + 0.65\, E_j\right)}_{\text{establishment prior}}$$

where

- $\bar{s}_j = \frac{1}{n_p}\sum_p s(p, j)$ is candidate $j$'s mean similarity across every prospect in the run. Subtracting it multiplicatively demotes "universal attractor" players who score well against everyone; because the factor is bounded in $(0, 1]$, a dissimilar player can never outrank a similar one.
- $E_j = \frac{1}{2}\left(\mathrm{pctile}(\text{career minutes}_j) + \mathrm{pctile}(\mathrm{APFV}_j)\right)$ is a career-establishment percentile. Among similarly shaped candidates this prefers the player with the more substantial career, correcting systematic pessimism without using draft position or any prospect-side post-draft data.

Selection then walks the ranking $R(p, \cdot)$ in descending order over the top $4 \times 10 = 40$ candidates subject to:

1. **Counterpart exclusion**: any NBA player whose normalized (diacritic- and punctuation-insensitive) name equals the prospect's is masked out, so the prospect's own NBA counterpart never appears.
2. **Era diversity**: at most 2 comps per career-era bucket (late-90s, early-2000s, late-2000s, early-2010s, late-2010s, 2020s, by career midpoint).
3. **Usage cap**: no NBA player may appear more than 2 times across all prospects in a single run (one draft class, or one current board). Earlier-processed prospects claim first; later ones fall to the next-best distinct comp. The era constraint is relaxed before the usage cap if a prospect runs short.

### 3.7 Displayed percentage

Composites are far below 1.0 on the $\exp(-d/0.08)$ scale, so the shared cosine-calibrated transform does not apply. Prospect comps use

$$\text{display} = 100 \cdot \left[\mathrm{clip}(s,\, 0,\, 1)\right]^{0.2},$$

which maps the stored-comp distribution to approximately 55–79 with a median near 69.
