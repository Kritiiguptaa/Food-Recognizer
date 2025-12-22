# 🍽️ Indian Food Image Classification (Punjabi & South Indian Cuisine)

An end-to-end deep learning system that classifies Indian dish images across Punjabi and South Indian cuisines, retrieves recipe metadata, and estimates nutritional values using USDA APIs. Trained using PyTorch EfficientNet models and deployed through a real-time Streamlit UI.

---

## ✨ Features

* 🔍 **Image-based food recognition** (Punjabi + South Indian dishes)
* 🤖 **EfficientNet-B0 & EfficientNet-B3 models** with transfer learning
* 🌐 **Real-time Streamlit web UI for inference**
* 📊 **Advanced PyTorch training pipeline** with augmentation & mixed precision
* 🍱 **Recipe lookup & metadata display from Excel databases**
* 🧮 **Nutritional estimation using USDA Food Data Central API**
* 🗂️ **Scripted dataset scraping & preparation**
* 📈 **Evaluation charts, confusion matrix & accuracy metrics**

---

## 🏗️ Tech Stack

**Languages:** Python 3.x
**Frameworks:** PyTorch, TorchVision, Streamlit
**Models:** EfficientNet-B0, EfficientNet-B3
**Data:** Pandas, Openpyxl, Excel Sheets
**Visualization:** Matplotlib, Seaborn
**Utilities:** Pillow, Requests, split-folders
**APIs:** USDA FoodData Central, SerpAPI, Bing Image Downloader

---

## 📁 Project Structure

```
app.py                        # Streamlit UI for inference
nutrition.py                 # USDA nutrition lookup
train_model.py              # Training scripts per cuisine
predict.py                  # CLI inference testing
graphs.py                   # Evaluation charts & metrics

cuisines/
 ├── Punjabi/               # Punjabi trained model + classes
 └── SouthIndian/           # South Indian trained model

data/
 ├── recipes.xlsx          # Per-cuisine recipe metadata 
 └── images/               # Training & test image datasets
```

---

## 🧠 Model Training Overview

### Punjabi Cuisine

* Model: EfficientNet-B0
* Training strategy:

  * Phase 1: classifier warm-up
  * Phase 2: unfreeze last MBConv blocks
* Augmentations: AutoAugment, RandomCrop, ColorJitter
* Optimizer: AdamW + Cosine LR
* Mixed-precision AMP enabled

### South Indian Cuisine

* Model: EfficientNet-B3
* Training strategy:

  * Phase 1: head only
  * Phase 2: full fine-tuning
* Augmentations: Flip, Crop, CutMix (0.1 prob.)

---

## 🎯 Model Performance

| Metric         | Punjabi | South Indian |
| -------------- | ------- | ------------ |
| Top-1 Accuracy | ~60%    | ~65%         |
| Top-3 Accuracy | ~85%    | ~88%         |

(TTA used in inference improves confidence reliability.)

---

## 🌐 Streamlit Application

**Run image classification & recipe lookup instantly.**

### Launch App

```bash
git clone <repo-url>
cd ML_MODEL

streamlit run app.py
```

### Features:

* Upload food image
* Display predicted dish + confidence ranking (Top-5)
* Show recipe steps, ingredients & URL
* Return nutrition profile (calories, protein, fat, sugar)

---

## 🔐 Environment Variables

Create `.env` file:

```
USDA_API_KEY = "YOUR_USDA_API_KEY"
SERPAPI_KEY = "YOUR_SERPAPI_KEY"
```

---

## 📦 Dependencies

Install requirements:

Core packages include:

* torch
* torchvision
* streamlit
* pandas
* matplotlib
* seaborn
* pillow
* requests

---

## 🧾 Dataset & Sources

* Images gathered using SerpAPI & Bing scrapers
* Split using `split-folders`
* Recipes sourced from internal dataset (Punjabi + South Indian)
* Cleaned using pandas scripts & excel transformations

---

## 📊 Evaluation Tools

`graphs.py` generates:

* Confusion matrices
* Per-class accuracy
* Most-confused pair visualization
* Confidence histograms

---

## 📌 Future Improvements

* Expand cuisines (Gujarati, Maharashtrian, Bengali etc.)
* ONNX model export for mobile deployment
* Faster inference using quantization
* Recipe recommendation engine
* Allergy nutrition tagging

---
