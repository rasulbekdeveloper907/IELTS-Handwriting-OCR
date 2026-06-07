# IELTS Handwriting OCR

This project uses Microsoft's TrOCR model
to recognize handwritten IELTS essays.

Features:

- TrOCR
- HuggingFace Transformers
- PyTorch
- Gradio
- Fine-tuning
- CER evaluation
- WER evaluation

Project Architecture 

IELTS Handwriting OCR
            │
            ▼
      Raw Images
            │
            ▼
       Roboflow Labeling
            │
            ▼
       train.csv
       valid.csv
       test.csv
            │
            ▼
      TrOCR Processor
            │
            ▼
     TrOCR Fine-Tuning
            │
            ▼
       CER/WER Metric
            │
            ▼
      Best Checkpoint
            │
            ▼
      Gradio Application
            │
            ▼
      HuggingFace Space