# Architecture

## Overview

This project answers one question by running two different pipelines
against the same images and questions, then comparing them head-to-head:

```
Image + Question
    |
    +---------------------------+
    |                           |
    v                           v
Direct VLM (vlm_qa/)      OCR + LLM (ocr_baseline/)
image + question -> VLM   image -> OCR -> extracted text -> LLM
    |                           |
    +---------------------------+
                |
                v
    Accuracy Comparison (evaluation/)
    graded against reference answers, broken down by image type
```

Both pipelines answer the exact same questions about the exact same
images, and both are graded against the same reference answers baked
into the synthetic dataset. That symmetry is what makes the comparison
meaningful -- neither pipeline gets an easier or harder question set.

---

## Stage 1: Dataset

`data/generate_sample_images.py` creates five synthetic images (bar
chart, line chart, pie chart, table, and a plain-text document image)
with known values, plus fifteen reference questions whose answers are
objectively checkable against those values.

Synthetic data instead of scraped images was a deliberate choice: every
reference answer is correct by construction, since the same script that
draws "Q3: $90k" onto the chart also writes "$90k" as the reference
answer. That removes any ambiguity in evaluation -- there's no need to
independently verify what the "true" answer to some found-online chart
actually was.

---

## Stage 2: Direct VLM Path

`vlm_qa/vlm_answer.py` sends the raw image, base64-encoded, straight to
a vision-capable model (Gemma 3's multimodal variants, or llava/qwen2.5-vl)
via Ollama's `/api/generate` endpoint, alongside the question as the
prompt. No text extraction step exists in this path at all -- the model
reasons directly over pixels.

---

## Stage 3: OCR + LLM Baseline

`ocr_baseline/ocr_then_llm.py` represents the traditional alternative:
Tesseract extracts whatever text it can find in the image, and that
extracted text -- not the image -- is handed to a text-only LLM call
along with the question.

This path exists specifically to make OCR error compounding visible and
measurable rather than anecdotal. If Tesseract misreads a number on a
chart's axis or garbles a word in the document image, the downstream LLM
call has no way to recover -- it never sees the original pixels, only
whatever text OCR handed it. Any failure introduced at the OCR stage
propagates silently into the final answer.

---

## Stage 4: Accuracy Comparison

`evaluation/compare_accuracy.py` grades both pipelines' answers against
the reference answers using an LLM-as-judge -- exact string matching
would be too strict, since "$90k" and "ninety thousand dollars" are the
same answer but wouldn't match character-for-character.

The comparison is broken down **by image type** (chart, table, document)
rather than reported as one blended number, because that breakdown is
the actual finding this project is built to produce. "Direct VLM: 80%,
OCR+LLM: 65%" is a summary. "OCR+LLM collapses on charts specifically,
where there's no literal text to extract, but performs closer to the VLM
on the plain-text document" is the explanation of *why* -- and it's only
visible once results are split by image type instead of averaged together.

---

## Why This Order

- Both answer pipelines (Stages 2-3) need the dataset (Stage 1) before
  they can run against anything.
- The accuracy comparison (Stage 4) needs completed answer sets from
  both pipelines -- there's no comparison possible with only one side.
- Splitting the final report by image type, rather than by pipeline
  alone, is what turns two accuracy percentages into an actual argument
  about *when* OCR-then-LLM is a reasonable design choice and *when* it
  quietly discards information a direct VLM would have used.

The point of running both pipelines side by side isn't to declare one
universally better -- it's to make the tradeoff between them concrete
and measurable instead of a matter of intuition.