from __future__ import annotations

from hiexp.evaluation import QaMetric, aggregate, normalize_answer_qa, score_one


def test_normalize_strips_articles_punct_case():
    assert normalize_answer_qa("The Lord of the Rings.") == "lord of rings"
    assert normalize_answer_qa("  A  CAT!!  ") == "cat"


def test_normalize_handles_non_string():
    assert normalize_answer_qa(None) == ""  # type: ignore[arg-type]


def test_score_em_exact():
    m = score_one("Skeleton Crew", ["Skeleton Crew"])
    assert m == QaMetric(em=1, cem=1, f1=1.0)


def test_score_cem_substring():
    m = score_one("The answer is Skeleton Crew, written in 2016.", "Skeleton Crew")
    assert m.em == 0
    assert m.cem == 1
    assert m.f1 > 0.0


def test_score_f1_partial_overlap():
    m = score_one("Lord Rings", "The Lord of the Rings")
    assert m.f1 > 0.0
    assert m.em == 0


def test_score_picks_max_over_multiple_golds():
    m = score_one("Paris", ["London", "Paris", "Tokyo"])
    assert m.em == 1
    assert m.cem == 1


def test_aggregate_mean():
    metrics = [
        QaMetric(em=1, cem=1, f1=1.0),
        QaMetric(em=0, cem=1, f1=0.5),
        QaMetric(em=0, cem=0, f1=0.0),
    ]
    out = aggregate(metrics)
    assert out["n"] == 3
    assert abs(out["em"] - 1 / 3) < 1e-9
    assert abs(out["cem"] - 2 / 3) < 1e-9
    assert abs(out["f1"] - 0.5) < 1e-9


def test_aggregate_empty():
    out = aggregate([])
    assert out == {"em": 0.0, "cem": 0.0, "f1": 0.0, "n": 0}
