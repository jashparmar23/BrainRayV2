# BrainRay

# 🧠 BrainRay – Medical Imaging + Local LLM Assistant

A dual-purpose AI tool that combines **deep learning-based brain tumor classification** with a **local language model (LLM) Q&A assistant**, built for enhanced diagnostics and knowledge support in medical applications.

> 🚧 This project was developed as part of an AI assignment for **Lanmentix**. A demo video will be uploaded soon.

---

## 🔍 Project Overview

**BrainRay** enables:
1. **Medical Image Classification**  
   Upload brain MRI scans and get predicted tumor type:
   - **Glioma**
   - **Meningioma**
   - **Pituitary**
   - **No Tumor**

2. **Local LLM-Powered Q&A Assistant**  
   Upload PDFs or text files and ask context-aware questions – answers are generated locally without internet access using lightweight models.

---

## 🧠 Models Used

### 1. CNN for MRI Classification
- Framework: **TensorFlow** + **Keras**
- Architecture: Custom Convolutional Neural Network
- Trained on: Brain MRI dataset with 4 classes

### 2. LLM Q&A Engine
- Base Model: **LLaMA 2** (loaded locally via `llama.cpp`)
- Vector Store: **FAISS**
- Embeddings: **HuggingFace Sentence Transformers**
- Orchestration: **LangChain**

---

## 🎮 Features

- 🖼️ Drag & drop MRI scans for instant classification
- 📄 Upload PDFs / text and ask questions with local LLM
- 🧠 Completely **offline** inference – no external API calls
- ⚡ Built with **Streamlit** for a smooth, fast UI

---

## 🛠️ Tech Stack

| Category              | Tools/Frameworks                               |
|-----------------------|------------------------------------------------|
| Deep Learning         | TensorFlow, Keras                              |
| LLM + Q&A             | llama.cpp, LangChain, FAISS, HuggingFace       |
| UI                    | Streamlit                                      |
| Image Preprocessing   | OpenCV, PIL                                    |
| Backend Language      | Python 3.10                                     |

---

## 🚀 Getting Started

### ✅ Prerequisites
Ensure Python 3.10 is installed.

### 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/jashparmar23/BrainRay.git
cd BrainRay

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
