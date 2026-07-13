"""
Grades both vlm_qa/vlm_answer.py's and ocr_baseline/ocr_then_llm.py's
answers against the reference answers in data/reference_qa.json, using
an LLM-as-judge to decide correctness (exact string matching would be
too strict -- "$90k" and "90 thousand dollars" are the same answer).

Produces per-method accuracy overall and broken down by image_type
(chart, table, document) -- the breakdown is the actual point of this
project: a single "VLM: 80%, OCR+LLM: 65%" number is interesting, but
"OCR+LLM drops to 40% on charts specifically, where there's no literal
text to extract" is the finding that explains *why*.

Usage:
    python compare_accuracy.py \
        --vlm_answers ../evaluation/results/vlm_answers.json \
        --ocr_answers ../evaluation/results/ocr_llm_answers.json \
        --judge_model gemma3:12b \
        --output ../evaluation/results/accuracy_report.json
"""

import argparse
import json
import re
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"

JUDGE_PROMPT_TEMPLATE = """You are grading whether a model's answer to a question is correct, given a known reference answer.

Question: {question}
Reference answer: {reference_answer}
Model's answer: {model_answer}

The model's answer doesn't need to match word-for-word -- it's correct if it conveys the same factual content as the reference answer. Reply with ONLY a JSON object, no other text:
{{"correct": true or false, "reasoning": "one short sentence"}}
"""


def call_judge(judge_model: str, question: str, reference_answer: str, model_answer: str, timeout: int = 60) -> dict:
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, reference_answer=reference_answer, model_answer=model_answer
    )

    payload = json.dumps({
        "model": judge_model,
        "prompt": judge_prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            raw_output = body.get("response", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        return {"correct": None, "reasoning": f"judge call failed: {e}"}

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return {"correct": None, "reasoning": "judge did not return parseable JSON"}

    try:
        parsed = json.loads(match.group(0))
        return {"correct": bool(parsed.get("correct")), "reasoning": parsed.get("reasoning", "")}
    except json.JSONDecodeError:
        return {"correct": None, "reasoning": "judge JSON was malformed"}


def grade_answers(answers: list, judge_model: str) -> list:
    graded = []
    for i, item in enumerate(answers, start=1):
        print(f"  [{i}/{len(answers)}] {item['question'][:60]}...")

        if item.get("error"):
            graded.append({**item, "correct": None, "grading_reasoning": "generation failed, not graded"})
            continue

        verdict = call_judge(judge_model, item["question"], item["reference_answer"], item["model_answer"])
        graded.append({**item, "correct": verdict["correct"], "grading_reasoning": verdict["reasoning"]})

    return graded


def summarize_by_type(graded: list, method_label: str) -> dict:
    by_type = {}
    for item in graded:
        img_type = item["image_type"]
        by_type.setdefault(img_type, {"correct": 0, "incorrect": 0, "ungraded": 0})

        if item["correct"] is True:
            by_type[img_type]["correct"] += 1
        elif item["correct"] is False:
            by_type[img_type]["incorrect"] += 1
        else:
            by_type[img_type]["ungraded"] += 1

    summary = {}
    for img_type, counts in by_type.items():
        graded_total = counts["correct"] + counts["incorrect"]
        accuracy = (counts["correct"] / graded_total * 100) if graded_total else 0.0
        summary[img_type] = {**counts, "accuracy_pct": round(accuracy, 1)}

    overall_correct = sum(c["correct"] for c in by_type.values())
    overall_graded = sum(c["correct"] + c["incorrect"] for c in by_type.values())
    overall_accuracy = (overall_correct / overall_graded * 100) if overall_graded else 0.0

    return {"method": method_label, "by_image_type": summary, "overall_accuracy_pct": round(overall_accuracy, 1)}


def main(args):
    with open(args.vlm_answers, "r") as f:
        vlm_answers = json.load(f)
    with open(args.ocr_answers, "r") as f:
        ocr_answers = json.load(f)

    print("Grading direct VLM answers...")
    vlm_graded = grade_answers(vlm_answers, args.judge_model)

    print("\nGrading OCR+LLM answers...")
    ocr_graded = grade_answers(ocr_answers, args.judge_model)

    vlm_summary = summarize_by_type(vlm_graded, "direct_vlm")
    ocr_summary = summarize_by_type(ocr_graded, "ocr_then_llm")

    report = {
        "vlm_summary": vlm_summary,
        "ocr_summary": ocr_summary,
        "vlm_graded_answers": vlm_graded,
        "ocr_graded_answers": ocr_graded,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Accuracy by Image Type ===")
    print(f"{'Type':<12} {'Direct VLM':<14} {'OCR+LLM':<14}")
    print("-" * 40)
    all_types = set(vlm_summary["by_image_type"].keys()) | set(ocr_summary["by_image_type"].keys())
    for img_type in sorted(all_types):
        vlm_acc = vlm_summary["by_image_type"].get(img_type, {}).get("accuracy_pct", "-")
        ocr_acc = ocr_summary["by_image_type"].get(img_type, {}).get("accuracy_pct", "-")
        print(f"{img_type:<12} {str(vlm_acc) + '%':<14} {str(ocr_acc) + '%':<14}")

    print(f"\nOverall -- Direct VLM: {vlm_summary['overall_accuracy_pct']}%, "
          f"OCR+LLM: {ocr_summary['overall_accuracy_pct']}%")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare direct VLM vs OCR+LLM accuracy against reference answers")
    parser.add_argument("--vlm_answers", default="results/vlm_answers.json")
    parser.add_argument("--ocr_answers", default="results/ocr_llm_answers.json")
    parser.add_argument("--judge_model", default="gemma3:12b")
    parser.add_argument("--output", default="results/accuracy_report.json")
    args = parser.parse_args()

    main(args)
