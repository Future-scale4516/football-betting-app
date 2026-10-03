"""
Team-name matching.

The model's team names come from football-data.co.uk ("Accrington",
"Man City", "Nott'm Forest"). Fixtures from anywhere else use different
spellings ("Accrington Stanley", "Manchester City"). Until now every
mismatch needed a hand-written entry in TEAM_NAME_MAPS, and a miss meant
the team was silently treated as a brand-new club with a guessed rating.

This matches names automatically, but conservatively — returning None
is always preferred to a wrong match, because a wrong match produces a
confident forecast for the wrong club.

Order of attempts:
  1. Explicit map entry (always wins — hand-checked beats clever)
  2. Exact match
  3. Same name after normalising case, accents, punctuation, "FC"/"AFC"
  4. One name is a whole-word prefix of the other, and exactly one known
     team qualifies ("Accrington Stanley" -> "Accrington")
  5. Very close spelling, and clearly closer than the runner-up
"""

import re
import unicodedata
from difflib import SequenceMatcher

# Purely decorative tokens. Deliberately short: words like "United",
# "City", "Real" and "Athletic" tell clubs apart (Bristol City vs Bristol
# Rovers, Real Madrid vs Atletico Madrid), so they are NOT stripped.
_NOISE = {"fc", "afc", "the"}


def normalise(name: str) -> str:
    """Lowercase, strip accents and punctuation, drop decorative tokens."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t not in _NOISE]
    return " ".join(tokens)


def match_team(name, known, explicit=None):
    """Returns the matching name from `known`, or None if there isn't a
    confident match.

    known    — iterable of team names the model has ratings for
    explicit — optional {other_source_name: model_name} overrides
    """
    known = list(known)
    if explicit and name in explicit and explicit[name] in known:
        return explicit[name]
    if name in known:
        return name

    n = normalise(name)
    if not n:
        return None
    norm_known = {k: normalise(k) for k in known}

    exact = [k for k, v in norm_known.items() if v == n]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None

    nt = n.split()
    candidates = []
    for k, v in norm_known.items():
        kt = v.split()
        if not kt:
            continue
        if nt[:len(kt)] == kt or kt[:len(nt)] == nt:
            candidates.append((min(len(kt), len(nt)), k))
    if candidates:
        longest = max(c[0] for c in candidates)
        best = [k for length, k in candidates if length == longest]
        return best[0] if len(best) == 1 else None

    scored = sorted(
        ((SequenceMatcher(None, n, v).ratio(), k) for k, v in norm_known.items()),
        reverse=True)
    if scored and scored[0][0] >= 0.88:
        if len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.05:
            return scored[0][1]
    return None


def match_fixture_names(fixtures, known, explicit=None):
    """fixtures: [(home, away, kickoff, started), ...] using another
    source's spellings. Returns (matched_fixtures, unmatched_names).

    Names that can't be matched are passed through unchanged — they get
    treated as new/unrated clubs downstream (provisional rating, clearly
    flagged), which is the right outcome for a genuinely promoted team —
    and are also reported so a wrong spelling can be spotted and added to
    TEAM_NAME_MAPS."""
    out, unmatched = [], []
    for home, away, ko, started in fixtures:
        h = match_team(home, known, explicit)
        a = match_team(away, known, explicit)
        if h is None:
            unmatched.append(home)
        if a is None:
            unmatched.append(away)
        out.append((h or home, a or away, ko, started))
    return out, sorted(set(unmatched))
