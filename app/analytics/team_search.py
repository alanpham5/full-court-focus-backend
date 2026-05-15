"""Team name search with ordered-character (subsequence) matching."""

from __future__ import annotations


def ordered_subsequence_score(query: str, text: str) -> float:
    """
  Score 0–100 when every query character appears in order in text.
  Consecutive matches at the start of text score highest.
  """
    q = query.strip().lower()
    t = text.lower()
    if not q or not t:
        return 0.0

    if t.startswith(q):
        return 100.0

    indices: list[int] = []
    qi = 0
    for i, ch in enumerate(t):
        if qi < len(q) and ch == q[qi]:
            indices.append(i)
            qi += 1

    if qi < len(q):
        return 0.0

    span = indices[-1] - indices[0] + 1
    compact = len(q) / span
    start_factor = 1.0 - (indices[0] / max(len(t), 1)) * 0.35
    return 100.0 * compact * start_factor


def score_team_match(query: str, meta: dict) -> float:
    """Best ordered-match score across abbreviation, full name, nickname, and name tokens."""
    q = query.strip()
    if not q:
        return 0.0

    scores: list[float] = []
    name = str(meta.get("name", ""))
    abbr = str(meta.get("abbreviation", ""))
    nickname = str(meta.get("nickname", ""))

    if abbr:
        abbr_l = abbr.lower()
        ql = q.lower()
        if abbr_l == ql:
            scores.append(100.0)
        elif abbr_l.startswith(ql):
            scores.append(98.0)
        else:
            scores.append(ordered_subsequence_score(q, abbr))

    for field in (name, nickname):
        if not field:
            continue
        scores.append(ordered_subsequence_score(q, field))
        for word in field.split():
            if word:
                scores.append(ordered_subsequence_score(q, word))

    return max(scores) if scores else 0.0


def rank_team_search(
    query: str,
    metadata: dict[str, dict],
    *,
    limit: int = 6,
    score_cutoff: float = 35.0,
) -> list[str]:
    """Return team IDs sorted by match quality (best first)."""
    scored: list[tuple[float, str]] = []
    for tid, meta in metadata.items():
        score = score_team_match(query, meta)
        if score >= score_cutoff:
            scored.append((score, tid))

    scored.sort(
        key=lambda item: (
            -item[0],
            str(metadata[item[1]].get("name", "")).lower(),
        )
    )
    return [tid for _, tid in scored[:limit]]
