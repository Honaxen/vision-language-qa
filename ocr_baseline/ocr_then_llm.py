"""
Traditional two-stage baseline: extract text from the image with OCR
(Tesseract), then send that extracted text -- not the image -- to a
text-only LLM call to answer the question.

This exists specifically to make OCR error compounding visible. If OCR
misreads "Room 204" as "Room 2O4" or garbles a number in a chart's axis
labels, the LLM never sees the image and has no way to recover -- it can
only work with whatever text OCR handed it. Comparing this against
vlm_qa/vlm_answer.py's direct-image answers is what makes that failure
mode measurable instead of anecdotal.

Requires Tesseract installed on the system (not just the Python wrapper):
    brew install tesseract

Usage:
    python ocr_then_llm.py \
        --llm_model gemma3:12b \
        --qa_file ../data/reference_qa.json \
        --images_dir ../data/images \
        --output ../evaluation/results/ocr_llm_answers.json
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import pytesseract
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/generate"

ANSWER_PROMPT_TEMPLATE = """The following text was extracted via OCR from an image. It may contain OCR errors (misread characters, garbled numbers, missing spacing).

Extracted text:
---
{ocr_text}
---

Based only on this extracted text, answer the following question as concisely as possible. If the text doesn't contain enough information to answer confidently, say so.

Question: {question}
"""


def run_ocr(image_path: Path) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_string(image).strip()


def call_llm(model: str, prompt: str, timeout: int = 60) -> dict:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "answer": body.get("response", "").strip(),
                "latency_ms": round(elapsed_ms, 2),
                "error": None,
            }
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        return {"answer": "", "latency_ms": None, "error": str(e)}


def main(args):
    with open(args.qa_file, "r") as f:
        dataset = json.load(f)

    images_dir = Path(args.images_dir)
    results = []
    total_questions = sum(len(entry["questions"]) for entry in dataset)
    count = 0

    for entry in dataset:
        image_path = images_dir / entry["image"]
        print(f"\n{entry['image']} ({entry['image_type']})")

        ocr_text = run_ocr(image_path)
        print(f"  OCR extracted {len(ocr_text)} characters")

        for q in entry["questions"]:
            count += 1
            print(f"  [{count}/{total_questions}] {q['question']}")

            prompt = ANSWER_PROMPT_TEMPLATE.format(ocr_text=ocr_text, question=q["question"])
            outcome = call_llm(args.llm_model, prompt)

            results.append({
                "image": entry["image"],
                "image_type": entry["image_type"],
                "question": q["question"],
                "reference_answer": q["reference_answer"],
                "model_answer": outcome["answer"],
                "ocr_extracted_text": ocr_text,
                "latency_ms": outcome["latency_ms"],
                "error": outcome["error"],
                "method": "ocr_then_llm",
            })

            if outcome["error"]:
                print(f"      error: {outcome['error']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    error_count = sum(1 for r in results if r["error"])
    print(f"\nDone. {len(results) - error_count}/{len(results)} questions answered.")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer questions about images via OCR + text-only LLM")
    parser.add_argument("--llm_model", default="gemma3:12b")
    parser.add_argument("--qa_file", default="../data/reference_qa.json")
    parser.add_argument("--images_dir", default="../data/images")
    parser.add_argument("--output", default="../evaluation/results/ocr_llm_answers.json")
    args = parser.parse_args()

    main(args)
