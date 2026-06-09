# 🌙 Dreamweaver — Personalized Bedtime Stories

Dreamweaver is a premium, interactive web application that senses a child's emotions using multimodal AI and generates a personalized, illustrated bedtime story video to guide them into sweet dreams. 

By analyzing facial expressions and voice recordings, Dreamweaver dynamically adapts the story's theme, narration voice speed/tone, and illustrations to match the child's emotional state (Sad, Happy, or Neutral).

---

## ✨ Key Features

1. **Multimodal Emotion Recognition (Late Decision Fusion)**:
   - **Facial Emotion Recognition (FER)**: Uses a custom-trained **EfficientNet-B2** architecture with Test-Time Augmentation (TTA) to predict emotion from face images or real-time webcam captures.
   - **Speech Emotion Recognition (SER)**: Processes speech logs (using mel-spectrogram features, delta, and delta-double features) through an **ensemble of 3 AudioCNN models**.
   - **MLP Decision Fusion**: Merges predictions from both modalities using a learned Multi-Layer Perceptron (MLP) that factors in prior contexts and real-time model reliability scores.

2. **Emotion-Guided Storytelling**:
   - Integrated with **Gemini 2.5 Flash** to generate custom, 4-paragraph storybooks tailored to the kid's detected mood:
     - *Sad*: Deeply comforting, secure stories focusing on friendship and warmth.
     - *Happy*: Lighthearted, playful fantasy adventures.
     - *Neutral*: Rhythmic, cozy nature tales featuring soothing environmental descriptions.

3. **Multi-Engine Voice Narration (TTS)**:
   - Supports premium voice synthesis via **ElevenLabs API** (using kid-friendly storyteller voice models).
   - Features automatic, seamless fallback to free **Microsoft Edge TTS** neural voices with customized speech rates and pitches adjusted to the target emotion.

4. ** Fairytale Illustrations (Pollinations AI)**:
   - Generates matching fairy tale scenes using Pollinations AI.
   - Features sidebar integration for a custom API Key, customized `User-Agent` wrappers to bypass rate-limiting, and error-proof local image fallback generation.

5. **Subtitle Overlay (No Binary Dependencies)**:
   - Subtitles are dynamically burned directly onto the image slides using Pillow before MoviePy rendering. This removes any requirement for installing complex system binaries like **ImageMagick** on Windows.

---

## 📂 Repository Structure

```text
C:\Users\aadid\Desktop\Emotion Recognition\
 ├── models\                       # Model Weight Files (Ignored by Git)
 │    ├── facial\
 │    │    └── best_face_model_stage2_3class.pth
 │    ├── audio\
 │    │    ├── ensemble_audio_model.pth
 │    │    └── final_best_audio_model.pth
 │    └── fusion\
 │         └── fusion_bundle.pt
 ├── app.py                        # Streamlit Web Application Code
 ├── README.md                     # Documentation
 ├── .gitignore                    # Git file exclusions
 ├── requirements.txt              # Python packages dependencies list
 ├── 01_FER_Training_Testing.ipynb # FER Training Notebook
 ├── 02_SER_Training_Testing.ipynb # SER Training Notebook
 ├── 03_Multimodal_Testing_and_Evaluation.ipynb # Decision Fusion Evaluation
 └── 04_Personalized_Bedtime_Story_Generation.ipynb # Pipeline Demonstration
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/emotion-recognition-dreamweaver.git
cd emotion-recognition-dreamweaver
```

### 2. Download and Place Model Weights
Since the model weights are too heavy for GitHub, they are excluded in `.gitignore`. 
Download your trained weight files from Google Drive and place them in the correct directories:
- Place `best_face_model_stage2_3class.pth` inside `models/facial/`
- Place `ensemble_audio_model.pth` and `final_best_audio_model.pth` inside `models/audio/`
- Place `fusion_bundle.pt` inside `models/fusion/`

### 3. Install Python Dependencies
Install all required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Web App

Launch the application locally:
```bash
python -m streamlit run app.py
```

Streamlit will start a local server. Open the link below in your browser:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🎮 How to Use

1. **Configuration Sidebar**:
   - Input your **Gemini API Key** (required to write the custom story).
   - Enter your **ElevenLabs API Key** (optional, falls back to free Edge-TTS if empty).
   - Enter your **Pollinations API Key** (defaults to a configured key to ensure image downloads).
2. **Modalities Capture**:
   - **Face Capture**: Toggle between using your live webcam or uploading a JPG/PNG face image.
   - **Audio Voice**: Upload a short `.wav` voice recording of the child.
3. **Generate**:
   - Click **🚀 Analyze Emotion & Tell Bedtime Story**.
   - Review the final fused emotion output and read the custom story text on the right side.
   - Watch and download the finalized fairytale **Story Video** complete with subtitle overlays and voice narration!

---

## 📝 Authors & Credits
Developed as part of the Multimodal Emotion Recognition Bedtime Story Generation Project.
- Story Generation powered by **Google Gemini**.
- Image Generation powered by **Pollinations.ai**.
- Text-to-Speech powered by **ElevenLabs** and **Edge-TTS**.
