"""
Khobor - Spaced Repetition System (SRS) Module
==============================================
Implements the SuperMemo-2 (SM-2) algorithm for calculating optimal review
intervals for vocabulary and current affairs flashcards.

Exam context:
- CDS and GATE-style exams require retaining extensive factual lists and high-register
  vocabulary over 6-12 months.
- The SM-2 algorithm models memory decay and spaces repetitions dynamically:
  easy cards appear less frequently, while difficult cards recur until mastered.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional


def sm2_update(ease_factor: float, interval: int, repetitions: int, quality: int) -> Tuple[float, int, int]:
    """
    Core SuperMemo-2 (SM-2) algorithm update calculation.

    Parameters
    ----------
    ease_factor : float
        Difficulty multiplier (default starts at 2.5, minimum 1.3).
    interval : int
        Current review interval in days.
    repetitions : int
        Number of consecutive successful reviews (quality >= 3).
    quality : int
        Recall rating from 0 to 5:
        - 5: Perfect recall with zero hesitation.
        - 4: Correct recall after brief hesitation.
        - 3: Correct recall with serious difficulty.
        - 2: Incorrect response, but correct answer remembered upon reveal.
        - 1: Incorrect response, correct answer was vaguely familiar.
        - 0: Complete blackout / total unfamiliarity.
        < 3 is considered a fail (resets repetitions & interval).
        >= 3 is considered a pass (grows interval).

    Returns
    -------
    Tuple[float, int, int]
        Updated (ease_factor, interval, repetitions).
    """
    # Step 1: Check pass/fail threshold
    if quality < 3:
        # Failure: reset repetition count and reset interval to 1 day
        repetitions = 0
        interval = 1
    else:
        # Success: advance repetition schedule
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 6
        else:
            interval = round(interval * ease_factor)
        repetitions += 1

    # Step 2: Recalculate ease factor
    # Formula: EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    # Minimum ease factor is clamped to 1.3 to avoid interval stagnation.
    ease_factor = max(1.3, ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))

    return ease_factor, interval, repetitions


def get_initial_srs_state(start_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate initial SM-2 metadata for a newly extracted flashcard.

    Parameters
    ----------
    start_date : Optional[str]
        Initial review date formatted as 'YYYY-MM-DD'. If None, defaults to today.

    Returns
    -------
    Dict[str, Any]
        Dictionary with initial SM-2 parameters ready to write to Google Sheets.
    """
    today_str = start_date if start_date else datetime.now().strftime("%Y-%m-%d")
    return {
        "ease_factor": 2.5,          # Standard starting ease factor in SM-2
        "interval_days": 1,          # First review interval is 1 day
        "repetitions": 0,            # 0 completed reviews yet
        "next_review_date": today_str,# Due immediately / today
        "last_reviewed": "",         # Not yet reviewed
    }


def calculate_next_review_date(last_reviewed_date_str: str, interval_days: int) -> str:
    """
    Calculate the next review date given the last review date and interval.

    Parameters
    ----------
    last_reviewed_date_str : str
        Date of review in 'YYYY-MM-DD' format.
    interval_days : int
        Number of days until the next scheduled review.

    Returns
    -------
    str
        Next review date in 'YYYY-MM-DD' format.
    """
    try:
        last_date = datetime.strptime(last_reviewed_date_str, "%Y-%m-%d")
    except ValueError:
        last_date = datetime.now()
    next_date = last_date + timedelta(days=interval_days)
    return next_date.strftime("%Y-%m-%d")


def attach_srs_to_card(card_data: Dict[str, Any], initial_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Attach SM-2 tracking fields to an extracted vocabulary or fact card dictionary.

    Parameters
    ----------
    card_data : Dict[str, Any]
        Dictionary containing card content (e.g. word, meaning, or fact, relevance).
    initial_date : Optional[str]
        Starting date string 'YYYY-MM-DD'. Defaults to today.

    Returns
    -------
    Dict[str, Any]
        Card dictionary enriched with SRS fields.
    """
    srs_fields = get_initial_srs_state(initial_date)
    enriched = dict(card_data)
    enriched.update(srs_fields)
    return enriched
