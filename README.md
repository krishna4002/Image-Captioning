# 🧠🖼 Medical & Normal Image Captioning using CLIP + OpenRouter LLM

This project aims to **generate intelligent and human-like captions** for both **medical X-ray images** (like chest scans) and **normal everyday images** using a hybrid AI approach. It combines **CLIP** for visual understanding and a **language model (LLM)** accessed through **OpenRouter API** for natural language generation.

---

## What Problem Does It Solve?

- **Medical images (like X-rays) are hard to interpret for non-doctors.**  
  This system generates readable and useful summaries of what's in the scan.

- **Normal image captioning is often needed in search engines, alt text, and content management.**  
  This app also supports captioning of regular photographs or scenes.

---

## Key Features

Supports **two modes**:
- **Medical Mode**: For interpreting chest X-ray images
- **Normal Mode**: For captioning everyday photos (COCO dataset)

- Upload your own .jpg, .jpeg, or .png image  
- Uses **CLIP (OpenAI)** to extract visual features  
- Generates captions via **LLM hosted on OpenRouter**  
- Built with a clean and interactive **Streamlit UI**  
- API key is **safely stored inside code** (not exposed to users)  
- Helpful for both *education**, **health accessibility**, and **machine vision** research  

---

## How It Works

1. You upload an image.
2. The system uses **CLIP (Contrastive Language–Image Pretraining)** to convert the image into a high-dimensional embedding vector.
3. This vector is simulated as part of a natural language **prompt**.
4. The prompt is sent to a **powerful LLM via OpenRouter API**.
5. The LLM generates a rich and accurate **caption**.

---

## 🧾 Example Use Cases

| Mode         | Input Image             | AI Caption Output |
|--------------|--------------------------|--------------------|
| 🏥 Medical   | Chest X-ray of patient   | "There is mild cardiomegaly with interstitial markings indicating pulmonary edema." |
| 🏞 Normal    | Beach photo              | "A woman walking along the beach with waves crashing behind her." |

---

## Project Structure

```
project/
│
├── app.py                        # Main Streamlit app (final version)
│
├── models/
│   └── clip_encoder.py           # CLIP-based feature extractor
│
├── src/
│   ├── inference.py              # Calls OpenRouter API
│   └── dataset.py                # Custom loaders for IU X-Ray and COCO
│
├── data/
│   ├── iu_xray/
│   │   ├── iu_xray.json          # IU X-Ray captions
│   │   └── images/               # IU X-Ray image files
│   └── coco/
│       ├── captions.json         # COCO image captions
│       └── images/               # COCO image files
│
└── requirements.txt              # Python package requirements
```

---

## Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/image-captioning-medical-normal.git
cd image-captioning-medical-normal
```

2. **Install Python Dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare Datasets**

- **Medical (IU X-Ray)**:
  - Download images from Kaggle or OpenI
  - Place them in: data/iu_xray/images/
  - Ensure you also have iu_xray.json in data/iu_xray/

- **Normal (COCO Lite)**:
  - Download sample images + captions
  - Place them in: data/coco/images/ and data/coco/captions.json

4. **Set Your OpenRouter API Key**

Adding api key in `.env` file:

```python
API_KEY = "YOUR_OPENROUTER_API_KEY"
```

5. **Run the App**

```bash
streamlit run app.py
```

---

## Dependencies

```txt
torchvision
torch
transformers
openai
streamlit
Pillow
requests
dotenv
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## Who Can Use This?

**Doctors & Medical Students**: To quickly interpret X-ray findings.  
**Photographers & Creators**: Auto-generate alt-text or image captions.  
**Researchers & Students**: Learn how to combine vision and language models.  
**Developers**: Use this as a base to build captioning tools or educational apps.

---

## Learning Highlights

- Feature extraction with CLIP
- Prompt engineering for vision-language LLMs
- OpenRouter API usage for LLM access
- Image handling & caption pairing
- Dataset loaders for both medical and COCO images

---

## Future Improvements

- [ ] Real-time webcam integration  
- [ ] Multilingual captioning  
- [ ] Use full CLIP image embeddings instead of simulated text prompts  
- [ ] Voice output for captions (TTS integration)

---

## About the Author

This project was created with the goal of making **medical imaging more accessible** to the public and researchers. It also demonstrates how **vision and language AI can work together** in real-world applications.

> Built with ❤ and AI
