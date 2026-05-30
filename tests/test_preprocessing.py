"""Unit tests for the Preprocessor class (Task 3.3).

Validates: Requirements 2.1, 2.2, 2.3
"""

from __future__ import annotations

import pytest

from src.preprocessing import Preprocessor


@pytest.fixture
def preprocessor() -> Preprocessor:
    return Preprocessor()


# ---------------------------------------------------------------------------
# URL removal
# ---------------------------------------------------------------------------

def test_clean_removes_http_url(preprocessor: Preprocessor) -> None:
    result = preprocessor.clean("Check this out http://example.com today")
    assert "http://example.com" not in result
    assert "Check this out" in result
    assert "today" in result


def test_clean_removes_https_url(preprocessor: Preprocessor) -> None:
    result = preprocessor.clean("Visit https://example.com/path?q=1 for more")
    assert "https://" not in result
    assert "Visit" in result
    assert "for more" in result


def test_clean_removes_www_url(preprocessor: Preprocessor) -> None:
    result = preprocessor.clean("Go to www.example.com for details")
    assert "www.example.com" not in result
    assert "Go to" in result
    assert "for details" in result


# ---------------------------------------------------------------------------
# @mention stripping
# ---------------------------------------------------------------------------

def test_clean_removes_mention(preprocessor: Preprocessor) -> None:
    result = preprocessor.clean("Hello @alice how are you")
    assert "@alice" not in result
    assert "Hello" in result
    assert "how are you" in result


def test_clean_removes_multiple_mentions(preprocessor: Preprocessor) -> None:
    result = preprocessor.clean("@bob and @carol went to the store")
    assert "@bob" not in result
    assert "@carol" not in result
    assert "went to the store" in result


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------

def test_clean_collapses_multiple_spaces(preprocessor: Preprocessor) -> None:
    result = preprocessor.clean("hello   world")
    assert result == "hello world"


def test_clean_collapses_tabs(preprocessor: Preprocessor) -> None:
    result = preprocessor.clean("hello\t\tworld")
    assert result == "hello world"


def test_clean_collapses_newlines(preprocessor: Preprocessor) -> None:
    result = preprocessor.clean("hello\n\nworld")
    assert result == "hello world"


def test_clean_strips_leading_trailing_whitespace(preprocessor: Preprocessor) -> None:
    result = preprocessor.clean("  hello world  ")
    assert result == "hello world"


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------

def test_clean_empty_string_returns_empty(preprocessor: Preprocessor) -> None:
    assert preprocessor.clean("") == ""


def test_clean_text_becomes_empty_after_cleaning(preprocessor: Preprocessor) -> None:
    # Only a URL — after removal and strip, result should be empty
    result = preprocessor.clean("http://example.com")
    assert result == ""


def test_clean_only_mention_returns_empty(preprocessor: Preprocessor) -> None:
    result = preprocessor.clean("@onlymention")
    assert result == ""


# ---------------------------------------------------------------------------
# Punctuation handling
# ---------------------------------------------------------------------------

def test_clean_preserves_punctuation_by_default(preprocessor: Preprocessor) -> None:
    result = preprocessor.clean("Hello, world! How's it going?")
    assert "," in result
    assert "!" in result
    assert "?" in result


def test_clean_strips_punctuation_when_disabled() -> None:
    p = Preprocessor(preserve_punctuation=False)
    result = p.clean("Hello, world! How's it going?")
    assert "," not in result
    assert "!" not in result
    assert "?" not in result
    assert "Hello" in result
    assert "world" in result


# ---------------------------------------------------------------------------
# batch_clean
# ---------------------------------------------------------------------------

def test_batch_clean_returns_same_length(preprocessor: Preprocessor) -> None:
    texts = ["hello world", "http://example.com", "@user hi"]
    result = preprocessor.batch_clean(texts)
    assert len(result) == len(texts)


def test_batch_clean_empty_list_returns_empty_list(preprocessor: Preprocessor) -> None:
    assert preprocessor.batch_clean([]) == []


def test_batch_clean_passes_empty_strings_through(preprocessor: Preprocessor) -> None:
    result = preprocessor.batch_clean(["", "hello", ""])
    assert result[0] == ""
    assert result[2] == ""
    assert result[1] == "hello"
