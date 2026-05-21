"""
Percent-encoding handler.

Attackers obfuscate URLs by percent-encoding characters that browsers
will decode at request time. Common patterns:
  - https://example.com/%6c%6f%67%69%6e (decodes to /login)
  - %25%32%65  (double-encoded dot, .)
  - Mixing literal and encoded forms inside the same URL

This module decodes percent-encoding iteratively until the string is
stable or a depth limit is reached, then reports both the depth and
the decoded form. Both signals feed into features.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote


MAX_DECODE_ITERATIONS = 5


@dataclass(frozen=True)
class EncodingAnalysis:
    """Result of percent-encoding analysis."""

    original: str
    decoded: str
    decode_iterations: int      # 0 if no encoding present
    had_double_encoding: bool   # True if >1 round needed


def iterative_unquote(url: str) -> EncodingAnalysis:
    """
    Repeatedly percent-decode until stable.

    Stops at MAX_DECODE_ITERATIONS to prevent pathological inputs
    (an attacker could craft inputs designed to expand on decode).
    """
    if not url:
        return EncodingAnalysis(original="", decoded="", decode_iterations=0,
                                had_double_encoding=False)

    current = url
    iterations = 0
    for _ in range(MAX_DECODE_ITERATIONS):
        nxt = unquote(current)
        if nxt == current:
            break
        current = nxt
        iterations += 1

    return EncodingAnalysis(
        original=url,
        decoded=current,
        decode_iterations=iterations,
        had_double_encoding=iterations > 1,
    )
