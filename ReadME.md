# ✍️ IELTS Handwriting OCR

🚀 A complete end-to-end deep learning system for recognizing handwritten IELTS essays using **Microsoft TrOCR, HuggingFace Transformers, PyTorch, and Gradio**.

This project converts raw handwritten essay images into clean digital text using a modern transformer-based OCR pipeline and is deployed as a web application via HuggingFace Spaces.

---

## 🌟 Features

- 🤖 Microsoft TrOCR (Transformer-based OCR model)
- 🧠 HuggingFace Transformers integration
- ⚡ PyTorch training & fine-tuning pipeline
- 🎨 Gradio web app (upload image → text output)
- 🔧 Custom dataset fine-tuning (handwritten IELTS essays)
- 📊 CER (Character Error Rate) evaluation
- 📏 WER (Word Error Rate) evaluation
- 🗂️ Structured dataset pipeline (train / valid / test)
- 🚀 Deployment on HuggingFace Spaces

---

## 🏗️ Project Architecture

🖼️ Raw handwritten IELTS essay images are collected as input data.  
These images are then labeled using Roboflow to create structured text-image pairs.

The dataset is split into:

- 📁 train.csv (training data)
- 📁 valid.csv (validation data)
- 📁 test.csv (testing data)

Each file contains image paths and corresponding ground truth text.

---

Next, the data is processed using the **TrOCR Processor**, which:

- Resizes and normalizes images
- Converts images into tensor format
- Tokenizes text into model-readable format

---

Then, the **TrOCR model is fine-tuned** using PyTorch.  
During training, the model learns to map handwritten image patterns to correct text outputs.

---

After training, the model is evaluated using:

- ✏️ CER (Character Error Rate) → measures character-level mistakes  
- 📖 WER (Word Error Rate) → measures word-level mistakes  

These metrics help identify how accurately the model is performing.

---

The best-performing model is saved as a **checkpoint**, based on validation performance and lowest error rates.

---

Finally, the trained model is integrated into a **Gradio web application**, where users can:

- 📤 Upload handwritten IELTS essay images  
- ⚡ Get instant text predictions  
- 🧾 View extracted digital text output  

---

The final system is deployed on **HuggingFace Spaces**, making it accessible online for real-time OCR inference.

---

## 💡 Project Impact

This project demonstrates a full AI pipeline that:

- ✨ Automates handwritten essay digitization  
- ✨ Reduces manual transcription effort  
- ✨ Supports IELTS preparation systems  
- ✨ Combines Computer Vision + NLP + Deep Learning  
- ✨ Shows real-world deployment of transformer models  

---

## 🔥 Tech Stack

- Microsoft TrOCR (OCR Transformer model)  
- HuggingFace Transformers  
- PyTorch  
- Roboflow (data annotation)  
- Gradio (UI)  
- HuggingFace Spaces (deployment)  
- CER / WER evaluation metrics  

---

## 📌 Summary

This project is an end-to-end intelligent OCR system that transforms handwritten IELTS essays into structured digital text using state-of-the-art transformer models and deploys it as a production-ready web application.
