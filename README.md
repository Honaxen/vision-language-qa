# Vision-Language QA

Question-answering over images — charts, tables, and documents — comparing a direct vision-language model against a traditional OCR-then-LLM pipeline.

---

## What This Project Demonstrates

Every text project in this portfolio works on pure text. `computer-vision-fundamentals` works on pure images. This one sits at the intersection: asking natural-language questions about visual content and measuring which pipeline actually answers them correctly.

| Concern | Solution |
|---|---|
| Can a model answer questions about a chart or table directly? | Direct VLM question-answering (Gemma 3 vision / llava / qwen2.5-vl via Ollama) |
| Is OCR-then-LLM actually worse, or just different? | Traditional OCR + LLM pipeline as a baseline for comparison |
| Where does each approach actually win? | Head-to-head accuracy on a chart/table/document QA set, judged against reference answers |
| How much does OCR error compound downstream? | Same questions, same images, broken down by image type — not a blended average |

---

## Architecture

```
Image + Question
  ↓
  ├─→ Direct VLM        image + question → VLM → answer
  └─→ OCR + LLM          image → OCR → extracted text → LLM → answer
  ↓
Accuracy Comparison  →  graded against reference answers, by image type
```

---

## Project Structure

```
vision-language-qa/
├── data/
│   ├── generate_sample_images.py   — synthetic chart/table/document images + reference Q&A
│   ├── images/                     — generated images
│   └── reference_qa.json           — 15 reference questions with verifiable answers
├── vlm_qa/
│   └── vlm_answer.py                — direct image + question → VLM answer
├── ocr_baseline/
│   └── ocr_then_llm.py              — Tesseract OCR → text-only LLM answer
├── evaluation/
│   ├── compare_accuracy.py          — LLM-as-judge grading, broken down by image type
│   └── results/
├── tests/
│   └── test_vision_qa.py            — 7/7 passing
├── docs/
│   └── architecture.md
└── requirements.txt
```

---

## Getting Started

```bash
pip install -r requirements.txt
brew install tesseract
ollama serve
```

Gemma 3's vision-capable variants (12B and above) already handle images — no separate vision model pull needed if you're using `gemma3:12b`. If you'd rather use a dedicated vision model: `ollama pull llava`.

### 1. Generate the dataset

```bash
cd data
python generate_sample_images.py
cd ..
```

### 2. Answer questions with the direct VLM path

```bash
python vlm_qa/vlm_answer.py \
  --model gemma3:12b \
  --qa_file data/reference_qa.json \
  --images_dir data/images \
  --output evaluation/results/vlm_answers.json
```

### 3. Answer the same questions with the OCR + LLM baseline

```bash
python ocr_baseline/ocr_then_llm.py \
  --llm_model gemma3:12b \
  --qa_file data/reference_qa.json \
  --images_dir data/images \
  --output evaluation/results/ocr_llm_answers.json
```

### 4. Compare accuracy

```bash
python evaluation/compare_accuracy.py \
  --vlm_answers evaluation/results/vlm_answers.json \
  --ocr_answers evaluation/results/ocr_llm_answers.json \
  --judge_model gemma3:12b \
  --output evaluation/results/accuracy_report.json
```

Actual results from this run:
```
=== Accuracy by Image Type ===
Type         Direct VLM     OCR+LLM
----------------------------------------
chart        100.0%         33.3%
document     100.0%         100.0%
table        100.0%         100.0%

Overall -- Direct VLM: 100.0%, OCR+LLM: 60.0%
```

The gap is entirely concentrated in charts. On documents and tables — where the image actually contains literal text — OCR+LLM matches the direct VLM exactly. On charts, where the "data" is bars, slices, and line positions rather than text, OCR extracts fragments (axis labels, a title, maybe a legend) but nothing resembling the actual values, and the downstream LLM has no way to recover what was never captured.

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Stack

Python · Ollama (Gemma 3 vision / llava / qwen2.5-vl) · Tesseract OCR · pytest

---

## What I Learned

**Synthetic data made grading unambiguous.**
Because the same script that draws "Q3: $90k" onto the chart also writes "$90k" as the reference answer, there's never a question of what the "real" answer was. That removed an entire category of evaluation noise that scraped or found-online images would have introduced.

**OCR doesn't fail everywhere — it fails exactly where there's no text to extract.**
The measured result made this precise instead of anecdotal: OCR+LLM matched the direct VLM exactly on documents and tables (100% both), then collapsed to 33.3% on charts specifically. That's not OCR being generally unreliable — Tesseract read the axis labels and titles just fine. It's that a chart's actual information (which bar is tallest, what a slice's percentage is) was never text in the first place, so there was nothing for OCR to extract and nothing for the downstream LLM to reason about.

**Averaging by image type instead of overall was the actual point of this project.**
A single blended accuracy number would have hidden exactly the finding this project exists to surface: OCR+LLM does reasonably on plain-text documents (where there's real text to extract) and falls apart on charts (where the "text" — bars, slices, line positions — was never text to begin with).

**LLM-as-judge grading needs tolerance built in, not exact matching.**
"$90k" and "ninety thousand dollars" are the same answer. Exact string comparison would have scored the second wrong for no good reason — the judge prompt explicitly asks for factual equivalence, not surface-level matching.

**Keeping generation and grading in separate stages paid off again.**
The same pattern from `llm-safety-redteam` and `llm-preference-alignment`: answers get recorded first, judged second. That split meant grading logic could be tested against fixed inputs (mocked judge verdicts) without ever needing a live model call in the test suite.

---

## Related Projects

- [computer-vision-fundamentals](https://github.com/Honaxen/computer-vision-fundamentals) — the image fundamentals this project builds on
- [rag-evaluation-framework](https://github.com/Honaxen/rag-evaluation-framework) — the same evaluation/reporting pattern, applied to visual QA instead of retrieval
- [llm-safety-redteam](https://github.com/Honaxen/llm-safety-redteam) — same LLM-as-judge grading approach

---

## Author

[Honaxen](https://github.com/Honaxen)