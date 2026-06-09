import os
import re
import gc
import urllib.parse
import asyncio
import json
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import cv2
import librosa
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import google.generativeai as genai
import edge_tts
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip
from gradio_client import Client, handle_file

# ============================================================
# Streamlit Styling (Premium Kid-Friendly Theme)
# ============================================================
st.set_page_config(
    page_title="Dreamweaver — Personalized Bedtime Stories",
    page_icon="🌙",
    layout="wide"
)

st.markdown("""
<style>
    .reportview-container {
        background: #1E1E2F;
    }
    .main {
        background: #11111E;
        color: #E2E2E2;
        font-family: 'Outfit', sans-serif;
    }
    h1 {
        color: #FFB703;
        font-weight: 700;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
    }
    h2, h3 {
        color: #2196F3;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6C63FF 0%, #3F3D56 100%);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        background: linear-gradient(135deg, #8B85FF 0%, #4E4C6D 100%);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌙 Dreamweaver</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:#A0A0B0;'>A magical bedtime story generator that senses your child's emotions and guides them into sweet dreams.</p>", unsafe_allow_html=True)

# ============================================================
# Paths & Global Configurations
# ============================================================
BASE_DRIVE_PATH = "C:/Users/aadid/Desktop/Emotion Recognition"  # Local relative workspace
MODEL_PATH        = os.path.join(BASE_DRIVE_PATH, "models")
FACIAL_MODEL_PATH = os.path.join(MODEL_PATH, "facial")
AUDIO_MODEL_PATH  = os.path.join(MODEL_PATH, "audio")
FUSION_MODEL_PATH = os.path.join(MODEL_PATH, "fusion")

# Default Parameters
SR           = 22050
N_MELS       = 128
FIXED_LENGTH = 3
SAMPLES      = SR * FIXED_LENGTH
EPSILON      = 1e-6
COMMON_EMOTIONS = ["Happy", "Neutral", "Sad"]
NUM_CLASSES     = len(COMMON_EMOTIONS)

MIN_RELIABILITY       = 0.15
CONFLICT_THRESHOLD    = 0.60
FACE_TEMPERATURE      = 1.2
AUDIO_TEMPERATURE     = 1.0
UNCERTAINTY_THRESHOLD = 1.10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Model Architectures
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SEBlock(channels)
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + identity)

class AudioCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm2d(32)
        self.res1   = ResidualBlock(32)
        self.pool1  = nn.MaxPool2d(2)
        self.drop1  = nn.Dropout(0.2)
        self.conv2  = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm2d(64)
        self.res2   = ResidualBlock(64)
        self.pool2  = nn.MaxPool2d(2)
        self.drop2  = nn.Dropout(0.3)
        self.conv3  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3    = nn.BatchNorm2d(128)
        self.pool3  = nn.MaxPool2d(2)
        self.drop3  = nn.Dropout(0.3)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc1    = nn.Linear(128, 64)
        self.drop_fc = nn.Dropout(0.4)
        self.fc2    = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.drop1(self.pool1(self.res1(F.relu(self.bn1(self.conv1(x))))))
        x = self.drop2(self.pool2(self.res2(F.relu(self.bn2(self.conv2(x))))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.gap(x).view(x.size(0), -1)
        x = self.drop_fc(F.relu(self.fc1(x)))
        return self.fc2(x)

class FusionMLP(nn.Module):
    def __init__(self, input_dim=11, hidden=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# Caching Model Loading Resource
# ============================================================
@st.cache_resource
def load_all_models():
    # 1. Face Model
    face_model_path = os.path.join(FACIAL_MODEL_PATH, "best_face_model_stage2_3class.pth")
    if not os.path.exists(face_model_path):
        return None, None, None, f"FER weights not found: {face_model_path}"
    
    face_model = models.efficientnet_b2(weights=None)
    in_features = face_model.classifier[1].in_features
    face_model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )
    face_model.load_state_dict(torch.load(face_model_path, map_location=device, weights_only=True))
    face_model = face_model.to(device).eval()

    # 2. Audio Model List (Ensemble or single seed)
    ensemble_bundle_path = os.path.join(AUDIO_MODEL_PATH, "ensemble_audio_model.pth")
    single_model_path    = os.path.join(AUDIO_MODEL_PATH, "final_best_audio_model.pth")
    audio_models_list = []

    if os.path.exists(ensemble_bundle_path):
        state_dicts = torch.load(ensemble_bundle_path, map_location=device, weights_only=True)
        for sd in state_dicts:
            m = AudioCNN(num_classes=NUM_CLASSES)
            m.load_state_dict(sd)
            m.to(device).eval()
            audio_models_list.append(m)
    elif os.path.exists(single_model_path):
        m = AudioCNN(num_classes=NUM_CLASSES)
        m.load_state_dict(torch.load(single_model_path, map_location=device, weights_only=True))
        m.to(device).eval()
        audio_models_list.append(m)
    else:
        return None, None, None, "SER weights not found."

    # 3. Fusion Model
    fusion_bundle_path = os.path.join(FUSION_MODEL_PATH, "fusion_bundle.pt")
    if not os.path.exists(fusion_bundle_path):
        return None, None, None, f"Fusion bundle not found: {fusion_bundle_path}"
    
    bundle = torch.load(fusion_bundle_path, map_location=device, weights_only=True)
    fusion_model = FusionMLP(input_dim=bundle["fusion_input_dim"], hidden=bundle["fusion_hidden"], num_classes=bundle["num_classes"])
    fusion_model.load_state_dict(bundle["fusion_mlp_state"])
    fusion_model = fusion_model.to(device).eval()

    # Haar Cascade detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    return face_model, audio_models_list, fusion_model, face_detector

# ============================================================
# Inference Functions
# ============================================================
def compute_face_quality(gray_face):
    blur = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    brightness = np.mean(gray_face)
    blur_norm = min(blur / 1000.0, 1.0)
    bright_norm = 1.0 - abs(brightness - 127) / 127.0
    return float(np.clip(0.7 * blur_norm + 0.3 * bright_norm, 0.0, 1.0))

def run_face_prediction(cv_image, face_model, face_detector):
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return {"valid": False}

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    face_crop = cv_image[y:y + h, x:x + w]
    quality = compute_face_quality(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY))

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    pil_face = Image.fromarray(face_rgb)

    face_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TTA transforms on PIL Image
    TTA_TFMS = [
        lambda img: face_transform(img),
        lambda img: face_transform(img.transpose(Image.FLIP_LEFT_RIGHT)),
        lambda img: face_transform(img.resize((232, 232)).crop((4, 4, 228, 228))),
        lambda img: face_transform(transforms.ColorJitter(brightness=0.1, contrast=0.1)(img))
    ]

    tta_probs = []
    with torch.inference_mode():
        for tfm in TTA_TFMS:
            t = tfm(pil_face).unsqueeze(0).to(device)
            out = face_model(t)
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
            tta_probs.append(prob)

    face_probs = np.mean(tta_probs, axis=0)
    face_probs = temperature_scale(face_probs, FACE_TEMPERATURE)

    pred_idx = int(np.argmax(face_probs))
    confidence = float(np.max(face_probs))
    reliability = 0.7 * confidence + 0.3 * quality
    entropy = prediction_entropy(face_probs)

    return {
        "emotion": COMMON_EMOTIONS[pred_idx],
        "probabilities": face_probs,
        "confidence": confidence,
        "quality": quality,
        "reliability": reliability,
        "entropy": entropy,
        "valid": True
    }

def run_audio_prediction(audio_bytes, audio_models_list):
    try:
        # Load audio using librosa from bytes
        # Save temp file
        temp_wav = "temp_st_audio.wav"
        with open(temp_wav, "wb") as f:
            f.write(audio_bytes)

        y, sr = librosa.load(temp_wav, sr=SR)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

        rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
        quality = min(rms / 0.1, 1.0)

        # Silence trimming
        y, _ = librosa.effects.trim(y, top_db=20)
        # Normalization
        y = librosa.util.normalize(y)

        if len(y) < SAMPLES:
            y = np.pad(y, (0, SAMPLES - len(y)))
        else:
            y = y[:SAMPLES]

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, win_length=2048, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        delta  = librosa.feature.delta(mel_db)
        delta2 = librosa.feature.delta(mel_db, order=2)

        stacked = np.stack([mel_db, delta, delta2], axis=0)
        for i in range(3):
            stacked[i] = (stacked[i] - stacked[i].mean()) / (stacked[i].std() + EPSILON)

        tensor = torch.tensor(stacked).unsqueeze(0).to(device)

        with torch.inference_mode():
            ensemble_probs = None
            for m in audio_models_list:
                p = torch.softmax(m(tensor), dim=1)
                ensemble_probs = p if ensemble_probs is None else ensemble_probs + p
            probs = (ensemble_probs / len(audio_models_list)).cpu().numpy()[0]

        audio_probs = temperature_scale(probs, AUDIO_TEMPERATURE)

        pred_idx = int(np.argmax(audio_probs))
        confidence = float(np.max(audio_probs))
        reliability = 0.6 * confidence + 0.4 * quality
        entropy = prediction_entropy(audio_probs)

        return {
            "emotion": COMMON_EMOTIONS[pred_idx],
            "probabilities": audio_probs,
            "confidence": confidence,
            "quality": quality,
            "reliability": reliability,
            "entropy": entropy,
            "valid": True
        }
    except Exception as e:
        return {"valid": False}

def temperature_scale(probs, temperature):
    log_p = np.log(np.clip(probs, 1e-9, 1.0))
    scaled = log_p / temperature
    shifted = scaled - np.max(scaled)
    exp_p = np.exp(shifted)
    return exp_p / exp_p.sum()

def prediction_entropy(probs):
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log2(probs)))

def fuse_emotions(face_res, audio_res, fusion_model):
    face_valid  = face_res.get("valid", False)
    audio_valid = audio_res.get("valid", False)

    if not face_valid and not audio_valid:
        return "Neutral", False, True, np.array([0.33, 0.34, 0.33])

    if face_valid and not audio_valid:
        return face_res["emotion"], False, face_res["entropy"] > UNCERTAINTY_THRESHOLD, face_res["probabilities"]

    if audio_valid and not face_valid:
        return audio_res["emotion"], False, audio_res["entropy"] > UNCERTAINTY_THRESHOLD, audio_res["probabilities"]

    # Both valid: Learned FusionMLP
    ctx_prior = np.ones(NUM_CLASSES) / NUM_CLASSES # Flat context prior in Streamlit stateless run

    feat_vec = np.concatenate([
        face_res["probabilities"],
        audio_res["probabilities"],
        ctx_prior,
        [max(face_res["reliability"], MIN_RELIABILITY),
         max(audio_res["reliability"], MIN_RELIABILITY)]
    ]).astype(np.float32)

    tensor = torch.tensor(feat_vec).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = fusion_model(tensor)
        fused_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    final_idx = int(np.argmax(fused_probs))
    final_emotion = COMMON_EMOTIONS[final_idx]
    entropy = prediction_entropy(fused_probs)
    uncertain = entropy > UNCERTAINTY_THRESHOLD

    conflict = (
        face_res["emotion"] != audio_res["emotion"]
        and face_res["confidence"] > CONFLICT_THRESHOLD
        and audio_res["confidence"] > CONFLICT_THRESHOLD
    )

    return final_emotion, conflict, uncertain, fused_probs

# ============================================================
# Main Page Setup
# ============================================================
st.sidebar.title("🛠️ Configuration")

gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Needed to dynamically generate the bedtime story. Leave blank to use a default story.")
elevenlabs_key = st.sidebar.text_input("ElevenLabs API Key", type="password", help="Optional. Leave blank to use free Microsoft Edge neural voices.")
pollinations_key = st.sidebar.text_input("Pollinations API Key", value="", type="password", help="Optional. Used to generate illustration images.")

# Load models
models_status = load_all_models()
if isinstance(models_status[-1], str) and "not found" in models_status[-1]:
    st.error(f"❌ Error: Model weights not found in shared drive folders. Please ensure you ran notebook cells 1-20 in Nb1 & Nb2. Error: {models_status[-1]}")
    st.stop()
else:
    face_model, audio_models_list, fusion_model, face_detector = models_status

# Initialize session state for keeping outputs
if "final_emotion" not in st.session_state:
    st.session_state.final_emotion = None
if "story_text" not in st.session_state:
    st.session_state.story_text = None
if "scene_prompts" not in st.session_state:
    st.session_state.scene_prompts = None
if "video_ready" not in st.session_state:
    st.session_state.video_ready = False
if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = None

# Grid Layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📸 Face Capture Modality")
    
    face_mode = st.radio("Face Input Select", ["Skip Face", "Use Webcam", "Upload Image File"], horizontal=True)
    
    cv_image = None
    if face_mode == "Use Webcam":
        webcam_file = st.camera_input("Smile for the camera!")
        if webcam_file:
            # Convert bytes to cv2 BGR image
            image = Image.open(webcam_file)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            st.success("Webcam photo captured successfully!")
    elif face_mode == "Upload Image File":
        upload_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if upload_img:
            image = Image.open(upload_img)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            st.image(image, caption="Uploaded Image", width=250)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🎤 Audio Voice Modality")
    
    audio_mode = st.radio("Audio Input Select", ["Skip Audio", "Upload Audio File"], horizontal=True)
    
    audio_bytes = None
    if audio_mode == "Upload Audio File":
        upload_aud = st.file_uploader("Upload audio WAV recording...", type=["wav"])
        if upload_aud:
            audio_bytes = upload_aud.read()
            st.audio(audio_bytes, format="audio/wav")
            st.success("Audio file loaded successfully!")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Analyze Emotion & Tell Bedtime Story"):
        with st.spinner("Processing child's emotion state..."):
            # 1. Unimodal predictions
            face_res = run_face_prediction(cv_image, face_model, face_detector) if cv_image is not None else {"valid": False}
            audio_res = run_audio_prediction(audio_bytes, audio_models_list) if audio_bytes is not None else {"valid": False}

            # 2. Decision Fusion
            emotion, conflict, uncertain, probs = fuse_emotions(face_res, audio_res, fusion_model)
            st.session_state.final_emotion = emotion

            # Display predictions
            st.info(f"✨ Detected Emotional State: **{emotion}**")
            if conflict:
                st.warning("⚠️ High conflict detected between face and voice signals!")
            if uncertain:
                st.warning("⚠️ Prediction uncertainty is high (unfamiliar inputs).")
            
            # 3. Generate story text
            if not gemini_key:
                st.session_state.story_text = "Once upon a time, a friendly little creature slept soundly on a cloud. It dreamt of bright colors, friendly owls, and warm starry nights. The moon shone softly, whispering sweet lullabies, and everything felt completely safe, warm, and happy."
                st.session_state.scene_prompts = [
                    "A friendly little cartoon creature sleeping on a cloud, moonlit sky",
                    "A cute owl sitting on a branch, detailed illustration",
                    "Beautiful yellow stars glowing in a dark blue sky, soft lighting",
                    "A happy children bedtime scene, cozy, cartoon drawing"
                ]
            else:
                try:
                    genai.configure(api_key=gemini_key)
                    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    prompt = f"""Write a soothing bedtime story for a child whose detected emotion is {emotion}.
                    - If emotion is Sad: Make it a deeply comforting, secure story about friendship, warmth, and sleeping peacefully.
                    - If emotion is Happy: Make it a fun, gentle, playful fantasy adventure.
                    - If emotion is Neutral: Make it a relaxing, rhythmic tale focusing on cozy nature sounds.
                    
                    Write exactly 4 paragraphs (around 200 words).
                    Also provide exactly 4 short scene illustration prompts (one for each paragraph) for Pollinations AI.
                    Return your response strictly in the following JSON format:
                    {{
                      "story": "...story text...",
                      "scenes": ["scene 1 description", "scene 2 description", "scene 3 description", "scene 4 description"]
                    }}
                    """
                    response = gemini_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                    story_content = json.loads(response.text)
                    st.session_state.story_text = story_content["story"]
                    st.session_state.scene_prompts = story_content["scenes"]
                except Exception as e:
                    st.error(f"Gemini API Error: {e}")
                    st.session_state.story_text = "Fallback story due to API key errors."
                    st.session_state.scene_prompts = ["A cozy bedtime illustration"]

            # 4. Generate TTS Narration & Pollinations Visuals & Video
            with st.spinner("Synthesizing narration and painting scenes..."):
                narration_file = "temp_st_narration.mp3"
                
                # TTS
                tts_used = None
                if elevenlabs_key:
                    voice_id = "EXAVITQu4vr4xnSDxMaL"
                    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                    headers = {"xi-api-key": elevenlabs_key, "Content-Type": "application/json"}
                    data = {
                        "text": st.session_state.story_text,
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {"stability": 0.35, "similarity_boost": 0.85, "style": 0.6, "use_speaker_boost": True}
                    }
                    response = requests.post(url, json=data, headers=headers)
                    if response.status_code == 200:
                        with open(narration_file, 'wb') as f:
                            f.write(response.content)
                        tts_used = "ElevenLabs API"
                    else:
                        st.sidebar.warning(f"⚠️ ElevenLabs API returned status {response.status_code}. Falling back to Microsoft Edge TTS.")
                        elevenlabs_key = "" # Fallback trigger
                
                if not elevenlabs_key:
                    # Edge TTS voice config
                    if emotion.lower() == "happy":
                        voice, rate, pitch = "en-US-GuyNeural", "+12%", "+4Hz"
                    elif emotion.lower() == "sad":
                        voice, rate, pitch = "en-US-JennyNeural", "-18%", "-4Hz"
                    else:
                        voice, rate, pitch = "en-US-AriaNeural", "+0%", "+0Hz"

                    communicate = edge_tts.Communicate(
                        text=st.session_state.story_text,
                        voice=voice,
                        rate=rate,
                        pitch=pitch
                    )
                    asyncio.run(communicate.save(narration_file))
                    tts_used = "Microsoft Edge TTS (Free Neural Voices)"
                
                st.session_state.tts_engine = tts_used

                # Image generation
                image_files = []
                for i, scene in enumerate(st.session_state.scene_prompts):
                    p_prompt = f"cinematic fairytale cartoon illustration, {scene}, emotional lighting, high resolution, cozy"
                    encoded_prompt = urllib.parse.quote(p_prompt)
                    
                    if pollinations_key.strip():
                        url = f"https://gen.pollinations.ai/image/{encoded_prompt}?key={pollinations_key.strip()}"
                    else:
                        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
                    
                    img_data = None
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                            }
                            res = requests.get(url, headers=headers, timeout=30)
                            if res.status_code == 200 and len(res.content) > 1000:
                                img_data = res.content
                                break
                        except Exception as e:
                            pass
                        if attempt < max_retries - 1:
                            time.sleep(2)
                    
                    if img_data is None:
                        # Fallback: create a solid color image with scene description text
                        from PIL import Image, ImageDraw
                        img = Image.new('RGB', (1024, 768), color=(30, 30, 50))
                        d = ImageDraw.Draw(img)
                        # Write the scene text
                        d.text((50, 350), f"Scene {i+1}:\n{scene}", fill=(255, 255, 255))
                        import io
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_data = img_byte_arr.getvalue()
                        st.warning(f"⚠️ Failed to download online image for Scene {i+1}. Using a placeholder.")

                    filename = f"st_scene_{i+1}.png"
                    with open(filename, 'wb') as f:
                        f.write(img_data)
                    image_files.append(filename)

                # MoviePy assembly
                audio_clip = AudioFileClip(narration_file)
                total_duration = audio_clip.duration
                slide_duration = total_duration / len(image_files)

                # Split story by double newlines to match slide count
                raw_paragraphs = [p.strip() for p in st.session_state.story_text.split("\n\n") if p.strip()]
                if len(raw_paragraphs) == 0:
                    raw_paragraphs = [st.session_state.story_text]
                
                paragraphs = []
                for i in range(len(image_files)):
                    if i < len(raw_paragraphs):
                        paragraphs.append(raw_paragraphs[i])
                    else:
                        paragraphs.append("")

                # Draw subtitles directly on the static images
                for img, text in zip(image_files, paragraphs):
                    if not text:
                        continue
                    try:
                        from PIL import Image, ImageDraw, ImageFont
                        pil_img = Image.open(img).convert("RGB")
                        width, height = pil_img.size
                        
                        # Word wrap text
                        words = text.split()
                        lines = []
                        current_line = []
                        for word in words:
                            current_line.append(word)
                            if len(" ".join(current_line)) > 55:
                                current_line.pop()
                                lines.append(" ".join(current_line))
                                current_line = [word]
                        if current_line:
                            lines.append(" ".join(current_line))
                        
                        try:
                            font = ImageFont.truetype("arial.ttf", 20)
                        except:
                            font = ImageFont.load_default()
                            
                        box_height = len(lines) * 26 + 30
                        
                        # Draw semi-transparent background box at the bottom
                        overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
                        draw_overlay = ImageDraw.Draw(overlay)
                        draw_overlay.rectangle(
                            [(0, height - box_height), (width, height)],
                            fill=(0, 0, 0, 160)
                        )
                        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
                        
                        # Draw text lines
                        draw = ImageDraw.Draw(pil_img)
                        text_y = height - box_height + 15
                        for line in lines:
                            try:
                                w = draw.textlength(line, font=font) if hasattr(draw, 'textlength') else len(line) * 10
                                text_x = (width - w) / 2
                            except:
                                text_x = 40
                            draw.text((text_x, text_y), line, font=font, fill=(255, 255, 255))
                            text_y += 26
                            
                        pil_img.save(img)
                    except Exception as e:
                        st.warning(f"Could not overlay subtitles on {img}: {e}")

                clips = []
                svd_clips = []  # Keep empty list to prevent downstream references from failing
                for img in image_files:
                    clip = ImageClip(img).set_duration(slide_duration).resize(height=720)
                    clips.append(clip)

                base_video = concatenate_videoclips(clips, method="compose")
                final_video = base_video.set_audio(audio_clip)

                final_video_path = "streamlit_story_video.mp4"
                final_video.write_videofile(
                    final_video_path,
                    fps=24,
                    codec="libx264",
                    audio_codec="aac",
                    verbose=False,
                    logger=None
                )

                audio_clip.close()
                final_video.close()
                for c in svd_clips:
                    try:
                        c.close()
                    except:
                        pass

                # Clean temp images & sound
                for f in image_files:
                    if os.path.exists(f): os.remove(f)
                if os.path.exists(narration_file): os.remove(narration_file)

                st.session_state.video_ready = True
                st.balloons()

with col_right:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📖 Personalized Bedtime Story Output")

    if st.session_state.final_emotion:
        st.markdown(f"**Emotion Guided Theme**: `{st.session_state.final_emotion}`")
        if st.session_state.tts_engine:
            st.markdown(f"**Voice Narration**: `{st.session_state.tts_engine}`")
        st.write(st.session_state.story_text)
        
        if st.session_state.video_ready and os.path.exists("streamlit_story_video.mp4"):
            st.subheader("🎥 Story Video with Voice & Subtitles")
            st.video("streamlit_story_video.mp4")
            
            with open("streamlit_story_video.mp4", "rb") as f:
                st.download_button(
                    label="💾 Download Bedtime Story Video",
                    data=f,
                    file_name=f"dreamweaver_story_{st.session_state.final_emotion.lower()}.mp4",
                    mime="video/mp4"
                )
    else:
        st.info("Input your face image/webcam feed and/or audio wav recording on the left, then click the button to generate your dynamic story video!")

    st.markdown("</div>", unsafe_allow_html=True)
