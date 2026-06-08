"""core.citation — 인용 overlap 검증 (환각 인용 차단)."""

from core.citation.verify import (
    VerificationResult,
    extract_claims,
    overlap_score,
    strip_unverified,
    verify_citations,
)

__all__ = [
    "VerificationResult",
    "extract_claims",
    "overlap_score",
    "strip_unverified",
    "verify_citations",
]
