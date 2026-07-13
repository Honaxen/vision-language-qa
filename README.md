# Vision-Language QA

Work in progress -- this README is a placeholder and will be replaced once the project is complete.

Question-answering over images -- charts, tables, and documents -- comparing a direct vision-language model against a traditional OCR-then-LLM pipeline.

---

## What This Project Will Demonstrate

Every text project in this portfolio works on pure text. computer-vision-fundamentals works on pure images (classification, edge detection). This one sits at the intersection: asking natural-language questions about visual content and getting grounded answers back.

Concern -> Solution (planned)
- Can a model answer questions about a chart or table directly?   -> Direct VLM question-answering (llava / qwen2.5-vl via Ollama)
- Is OCR-then-LLM actually worse, or just different?               -> Traditional OCR + LLM pipeline as a baseline for comparison
- Where does each approach actually win?                            -> Head-to-head accuracy on a chart/document QA set, judged against reference answers
- How much does OCR error compound downstream?                      -> Directly comparable failure cases between the two pipelines

---

## Planned Architecture

Image + Question
  -> Path A: Direct VLM (vlm_qa/)          image + question -> VLM -> answer
  -> Path B: OCR Baseline (ocr_baseline/)  image -> OCR -> extracted text -> LLM -> answer
  -> Evaluation (evaluation/)              compare both answers against a reference answer, per question
  -> Accuracy Report                        VLM accuracy vs OCR+LLM accuracy, by image type

---

## Project Structure

vision-language-qa/
  data/             - sample images (charts, tables, documents) + reference Q&A
  vlm_qa/            - direct VLM question-answering
  ocr_baseline/      - OCR + LLM pipeline
  evaluation/        - accuracy comparison, LLM-as-judge grading
  tests/
  docs/

---

## Stack

Python - Ollama (llava / qwen2.5-vl) - Tesseract OCR - pytest

---

## Status

- [ ] Sample image + reference Q&A dataset (charts, tables, documents)
- [ ] Direct VLM question-answering pipeline
- [ ] OCR + LLM baseline pipeline
- [ ] Accuracy evaluation comparing both approaches

---

## Related Projects

- [computer-vision-fundamentals](https://github.com/Honaxen/computer-vision-fundamentals) -- the image fundamentals this project builds on
- [rag-evaluation-framework](https://github.com/Honaxen/rag-evaluation-framework) -- similar evaluation/reporting pattern, applied to visual QA instead of retrieval

---

## Author

[Honaxen](https://github.com/Honaxen)
