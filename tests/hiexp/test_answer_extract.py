from __future__ import annotations

from hiexp.evaluation import extract_boxed_answer, extract_exact_answer


def test_extract_boxed_simple():
    assert extract_boxed_answer(r"some text \boxed{Skeleton Crew} more") == "Skeleton Crew"


def test_extract_boxed_missing():
    assert extract_boxed_answer("no box here") == ""


def test_extract_boxed_last_wins():
    assert extract_boxed_answer(r"\boxed{first} and \boxed{last}") == "last"


def test_extract_exact_answer_simple():
    text = (
        "Explanation: The MacArthur fellow in 2018 was Dominique Morisseau [12].\n"
        "Exact Answer: Skeleton Crew\n"
        "Confidence: 88%"
    )
    assert extract_exact_answer(text) == "Skeleton Crew"


def test_extract_exact_answer_bolded():
    text = (
        "**Explanation:** ...\n"
        "**Exact Answer:** Paris\n"
        "**Confidence:** 95%"
    )
    assert extract_exact_answer(text) == "Paris"


def test_extract_exact_answer_no_confidence():
    text = "Exact Answer: 42"
    assert extract_exact_answer(text) == "42"


def test_extract_exact_answer_missing():
    assert extract_exact_answer("no exact answer here") == ""
