# bubble_omr/scoring_defaults.py
from dataclasses import dataclass

@dataclass(frozen=True)
class ScoringDefaults:
    # Single source of truth for scoring thresholds
    min_fill: float = 0.20          # minimum filled fraction to accept non-blank
    top2_ratio: float = 0.80        # second-best must be <= top2_ratio * best
    min_score: float = 10.0         # absolute gap in percentage points (100*(best-second))
    min_abs: float = 0.10           # absolute minimum fill guard in _pick_single_from_scores

DEFAULTS = ScoringDefaults()

def apply_overrides(
    min_fill: float | None = None,
    top2_ratio: float | None = None,
    min_score: float | None = None,
    min_abs: float | None = None,
) -> ScoringDefaults:
    # produce an overridden immutable config without mutating DEFAULTS
    return ScoringDefaults(
        min_fill = DEFAULTS.min_fill if min_fill is None else min_fill,
        top2_ratio = DEFAULTS.top2_ratio if top2_ratio is None else top2_ratio,
        min_score = DEFAULTS.min_score if min_score is None else min_score,
        min_abs = DEFAULTS.min_abs if min_abs is None else min_abs,
    )
