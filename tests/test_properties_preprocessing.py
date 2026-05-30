"""Property-based tests for Preprocessor (Task 3.2).

**Validates: Requirements 2.2**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.preprocessing import Preprocessor


# ---------------------------------------------------------------------------
# Property 11: Batch Clean Length Preservation
# ---------------------------------------------------------------------------

@given(st.lists(st.text(), min_size=0, max_size=50))
@settings(max_examples=50)
def test_batch_clean_length_preservation(texts: list[str]) -> None:
    """**Validates: Requirements 2.2**

    For any list of text strings, calling batch_clean() should return a list
    of the same length where each element equals clean() applied to the
    corresponding input.
    """
    preprocessor = Preprocessor()

    result = preprocessor.batch_clean(texts)

    assert len(result) == len(texts), (
        f"batch_clean() returned {len(result)} elements for input of length {len(texts)}"
    )

    for i, (original, cleaned) in enumerate(zip(texts, result)):
        expected = preprocessor.clean(original)
        assert cleaned == expected, (
            f"batch_clean()[{i}] = {cleaned!r} != clean({original!r}) = {expected!r}"
        )
