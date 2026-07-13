"""
Answers every question in data/reference_qa.json by sending the image
directly to a vision-language model via Ollama -- no OCR, no text
extraction step. The model sees the pixels and answers directly.

Ollama's /api/generate endpoint accepts images as a list of base64-encoded
strings alongside the prompt, which is what makes this path simpler than
the OCR baseline: one API call per question, no intermediate text stage.

Usage:
    python vlm_answer.py \
        --model llava \
        --qa_file ../data/reference_qa.json \
        --images_dir ../data/images \
        --output ../evaluation/results/vlm_answers.json

Requires a vision-capable model pulled in Ollama first, e.g.:
    ollama pull llava
    (or: ollama pull qwen2.5vl)
"""

import argparse
import base64
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm(model: str, question: str, image_b64: str, timeout: int = 90) -> dict:
    payload = json.dumps({
        "model": model,
        "prompt": question,
        "images": [image_b64],
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

        image_b64 = encode_image(image_path)

        for q in entry["questions"]:
            count += 1
            print(f"  [{count}/{total_questions}] {q['question']}")

            outcome = call_vlm(args.model, q["question"], image_b64)

            results.append({
                "image": entry["image"],
                "image_type": entry["image_type"],
                "question": q["question"],
                "reference_answer": q["reference_answer"],
                "model_answer": outcome["answer"],
                "latency_ms": outcome["latency_ms"],
                "error": outcome["error"],
                "method": "direct_vlm",
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
    parser = argparse.ArgumentParser(description="Answer questions about images directly with a VLM")
    parser.add_argument("--model", default="llava", help="Ollama vision model name (e.g. llava, qwen2.5vl)")
    parser.add_argument("--qa_file", default="../data/reference_qa.json")
    parser.add_argument("--images_dir", default="../data/images")
    parser.add_argument("--output", default="../evaluation/results/vlm_answers.json")
    args = parser.parse_args()

    main(args)
