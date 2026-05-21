"""
Unicode and punycode handling.

Phishing attackers exploit Unicode in three primary ways:
  1. Homograph attacks: using look-alike characters from other scripts
     (e.g., Cyrillic 'а' U+0430 vs Latin 'a' U+0061) to spoof domains.
  2. Punycode abuse: registering xn-- domains that render as look-alikes
     in browsers that don't display punycode for mixed scripts.
  3. Compatibility character abuse: ligatures, full-width forms, etc.

This module normalizes hostnames to canonical NFKC Unicode form,
decodes punycode, and flags mixed-script domains as suspicious.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import idna


# Unicode script ranges we care about for homograph detection.
# This is intentionally coarse; perfect script identification requires
# the unicodedata module's script property (Python 3.12+) or a library
# like uniseg. For phishing detection, coarse is sufficient.
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "latin": [(0x0041, 0x007A), (0x00C0, 0x024F)],
    "cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "greek": [(0x0370, 0x03FF)],
    "han": [(0x4E00, 0x9FFF)],
    "arabic": [(0x0600, 0x06FF)],
    "hebrew": [(0x0590, 0x05FF)],
}


@dataclass(frozen=True)
class UnicodeAnalysis:
    """Result of Unicode analysis on a hostname."""

    original: str
    decoded: str               # Punycode-decoded, NFKC-normalized
    had_punycode: bool
    is_mixed_script: bool
    scripts_detected: frozenset[str]
    has_invisible_chars: bool


def _detect_script(codepoint: int) -> str | None:
    """Return the script name for a codepoint, or None if not tracked."""
    for script, ranges in SCRIPT_RANGES.items():
        for start, end in ranges:
            if start <= codepoint <= end:
                return script
    return None


def _has_invisible_characters(text: str) -> bool:
    """Detect zero-width and invisible characters used to evade matching."""
    invisible_categories = {"Cf", "Mn"}  # Format and nonspacing marks
    invisible_chars = {
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\u2060",  # Word joiner
        "\ufeff",  # Zero-width no-break space
    }
    if any(c in text for c in invisible_chars):
        return True
    return any(unicodedata.category(c) in invisible_categories for c in text)


def decode_punycode(hostname: str) -> tuple[str, bool]:
    """
    Decode an IDNA/punycode hostname to its Unicode form.

    Returns (decoded_hostname, had_punycode). If the input had no
    xn-- labels, returns it unchanged with had_punycode=False.
    """
    if not hostname:
        return hostname, False

    had_punycode = "xn--" in hostname.lower()
    if not had_punycode:
        return hostname, False

    try:
        # idna.decode handles per-label decoding correctly
        decoded = idna.decode(hostname)
        return decoded, True
    except (idna.IDNAError, UnicodeError):
        # Malformed punycode: return original, the model will see
        # the suspicious xn-- prefix in the features.
        return hostname, True


def analyze_hostname(hostname: str) -> UnicodeAnalysis:
    """
    Perform full Unicode analysis on a hostname.

    The decoded form is what feature extraction should use for textual
    features; the original is preserved so we can flag the difference
    as a suspicious signal.
    """
    if not hostname:
        return UnicodeAnalysis(
            original="",
            decoded="",
            had_punycode=False,
            is_mixed_script=False,
            scripts_detected=frozenset(),
            has_invisible_chars=False,
        )

    decoded, had_punycode = decode_punycode(hostname)
    # NFKC folds compatibility characters (full-width, ligatures) to canonical
    normalized = unicodedata.normalize("NFKC", decoded)

    scripts: set[str] = set()
    for char in normalized:
        if char.isalpha():
            script = _detect_script(ord(char))
            if script is not None:
                scripts.add(script)

    # Mixed script is a strong homograph signal. Two or more scripts in
    # a single label is the canonical indicator.
    is_mixed = len(scripts) > 1

    return UnicodeAnalysis(
        original=hostname,
        decoded=normalized,
        had_punycode=had_punycode,
        is_mixed_script=is_mixed,
        scripts_detected=frozenset(scripts),
        has_invisible_chars=_has_invisible_characters(normalized),
    )
