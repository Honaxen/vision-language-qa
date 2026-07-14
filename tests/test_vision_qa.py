"""
Unit tests for the pure aggregation logic in evaluation/compare_accuracy.py:
summarize_by_type() and grade_answers()'s handling of judge verdicts and
generation errors. The actual call_judge() Ollama call is mocked --
what's being verified is the logic that combines verdicts into per-type
and overall accuracy, not the judge model's real behavior.

Same pattern as llm-safety-redteam and llm-preference-alignment: pure
decision/aggregation logic gets unit tests, live-model calls are a
manual/integration concern.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

import compare_accuracy  # noqa: E402


# --- summarize_by_type() tests ---

def make_graded_item(image_type, correct):
    return {
        "image_type": image_type,
        "question": "irrelevant for this test",
        "correct": correct,
    }


def test_summarize_computes_accuracy_per_type():
    graded = [
        make_graded_item("chart", True),
        make_graded_item("chart", True),
        make_graded_item("chart", False),
        make_graded_item("table", True),
    ]
    summary = compare_accuracy.summarize_by_type(graded, "direct_vlm")

    assert summary["by_image_type"]["chart"]["correct"] == 2
    assert summary["by_image_type"]["chart"]["incorrect"] == 1
    assert summary["by_image_type"]["chart"]["accuracy_pct"] == pytest.approx(66.7, abs=0.1)
    assert summary["by_image_type"]["table"]["accuracy_pct"] == 100.0


def test_summarize_excludes_ungraded_from_accuracy_denominator():
    graded = [
        make_graded_item("document", True),
        make_graded_item("document", None),  # ungraded -- generation failed
    ]
    summary = compare_accuracy.summarize_by_type(graded, "direct_vlm")

    # accuracy should be 100% (1/1), not 50% (1/2) -- the ungraded item
    # shouldn't silently count against the model
    assert summary["by_image_type"]["document"]["accuracy_pct"] == 100.0
    assert summary["by_image_type"]["document"]["ungraded"] == 1


def test_summarize_overall_accuracy_combines_all_types():
    graded = [
        make_graded_item("chart", True),
        make_graded_item("chart", False),
        make_graded_item("table", True),
        make_graded_item("table", True),
    ]
    summary = compare_accuracy.summarize_by_type(graded, "ocr_then_llm")

    # 3 correct out of 4 graded total
    assert summary["overall_accuracy_pct"] == 75.0


def test_summarize_handles_type_with_no_graded_items():
    graded = [make_graded_item("chart", None)]  # only ungraded items
    summary = compare_accuracy.summarize_by_type(graded, "direct_vlm")

    assert summary["by_image_type"]["chart"]["accuracy_pct"] == 0.0


# --- grade_answers() tests ---

@patch("compare_accuracy.call_judge")
def test_grade_answers_marks_correct_from_judge_verdict(mock_call_judge):
    mock_call_judge.return_value = {"correct": True, "reasoning": "matches reference"}

    answers = [{
        "image_type": "chart", "question": "q", "reference_answer": "90",
        "model_answer": "ninety", "error": None,
    }]

    graded = compare_accuracy.grade_answers(answers, "judge-model")
    assert graded[0]["correct"] is True
    assert graded[0]["grading_reasoning"] == "matches reference"


@patch("compare_accuracy.call_judge")
def test_grade_answers_skips_judging_on_generation_error(mock_call_judge):
    answers = [{
        "image_type": "table", "question": "q", "reference_answer": "12",
        "model_answer": "", "error": "connection refused",
    }]

    graded = compare_accuracy.grade_answers(answers, "judge-model")

    assert graded[0]["correct"] is None
    assert graded[0]["grading_reasoning"] == "generation failed, not graded"
    mock_call_judge.assert_not_called()


@patch("compare_accuracy.call_judge")
def test_grade_answers_handles_unparseable_judge_output(mock_call_judge):
    mock_call_judge.return_value = {"correct": None, "reasoning": "judge did not return parseable JSON"}

    answers = [{
        "image_type": "document", "question": "q", "reference_answer": "Room 204",
        "model_answer": "some garbled text", "error": None,
    }]

    graded = compare_accuracy.grade_answers(answers, "judge-model")
    assert graded[0]["correct"] is None
