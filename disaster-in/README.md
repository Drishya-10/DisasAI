# 🚨 Disaster-In-Phase — Emergency AI Call Dispatch System

> **A real-time, multi-language AI pipeline that transforms noisy emergency calls into structured, dispatched rescue operations.**

Part of the **DisasAI** project — an AI-powered disaster response intelligence platform.

---

## 📸 Overview

When someone calls in a crisis — a flood, fire, building collapse, gas leak, or road accident — this system:

1. 🎙️ **Receives** their noisy, panicked audio call
2. 🔊 **Cleans** the audio using a 3-stage AI denoising pipeline
3. 🗣️ **Understands** their speech in native Indian languages (Hindi, Tamil, Telugu, Kannada, Bengali)
4. 🌐 **Translates** it to English
5. 🤖 **Extracts** structured emergency intelligence via LLM
6. 🚒 **Routes** the right rescue units automatically
7. 👮 **Presents** everything to a human operator who confirms the dispatch

---

## 🏗️ Architecture

```
Noisy Emergency Call
       │
       ▼
Denoiser (Butterworth → Spectral Gating → GRU Neural Net)
       │
       ▼
Gnani ASR — Native speech → Native language text
       │
       ▼
Translation Engine — Native text → English text
       │
       ▼
LLM Intelligence — Classify · Severity · Extract · JSON
       │
       ▼
Dispatch Engine — Type + Severity → Rescue units to send
       │
       ▼
Human Operator Confirms — Review, edit, authorize dispatch
```

---

## 🌟 Features

- **Multi-language support** — Hindi, Tamil, Telugu, Kannada, Bengali, English
- **3-stage denoiser** — handles flood rain, fire crackle, highway traffic, industrial hiss
- **Gnani.ai ASR** — Indic language speech recognition with API integration
- **Offline fallback** — works 100% without internet or API keys
- **LLM intelligence** — Gemini / Groq / offline NLP engine (auto-fallback)
- **Tactical dispatch** — NDRF, USAR, Fire Brigade, Highway EMS, CBRN Hazmat routing
- **Human-in-the-loop** — operator console with override and confirmation
- **6 preset scenarios** — ready-to-test noisy emergency audio files
- **Incident audit log** — full history of all processed calls
- **Dark command-center UI** — military-grade dashboard

---

## 🗂️ Project Structure

```
disaster-in/
├── backend/
│   ├── app.py                  ← FastAPI server (main entry point)
│   ├── config.py               ← API keys configuration
│   ├── denoiser.py             ← Butterworth + Spectral Gating + GRU
│   ├── asr_service.py          ← Gnani.ai ASR integration
│   ├── translation_service.py  ← deep-translator / Google Translate
│   ├── llm_analyzer.py         ← Gemini / Groq / offline NLP
│   ├── dispatch_engine.py      ← Tactical unit routing engine
│   ├── models.py               ← Pydantic data schemas
│   └── sample_generator.py     ← Generates synthetic noisy test audio
│
├── frontend/
│   ├── index.html              ← Single-page command center dashboard
│   ├── css/style.css           ← Dark military-grade theme
│   └── js/app.js               ← Pipeline orchestration & UI logic
│
├── data/
│   └── samples/                ← 6 preset emergency WAV files
│
├── test_pipeline.py            ← End-to-end integration test
├── .gitignore
└── README.md
```

---

## 🔊 Denoising Pipeline (Stage 2)

| Stage | Method | Removes |
|-------|--------|---------|
| **Butterworth Filter** | Zero-phase 4th-order bandpass (100–3800 Hz) | Subsonic rumble, electrical hum, RF hiss |
| **Spectral Gating** | Dynamic noise floor tracking + soft sigmoid mask | Background noise, static, rain, wind |
| **GRU Neural Net** | Bidirectional recurrent temporal mask prediction | Burst noise, residual transients |

---

## 🗣️ Supported Languages

| Language | Code | ASR Provider |
|----------|------|-------------|
| Hindi (हिन्दी) | `hi` | Gnani.ai `hi-IN` |
| Tamil (தமிழ்) | `ta` | Gnani.ai `ta-IN` |
| Telugu (తెలుగు) | `te` | Gnani.ai `te-IN` |
| Kannada (ಕನ್ನಡ) | `kn` | Gnani.ai `kn-IN` |
| Bengali (বাংলা) | `bn` | Gnani.ai `bn-IN` |
| English | `en` | Gnani.ai `en-IN` |

---

## 🚒 Dispatch Coverage

| Emergency Type | Primary Agency | Key Units |
|---------------|---------------|-----------|
| 🌊 FLOOD | NDRF 9th Battalion | Rescue boats, diving teams, dewatering pumps |
| 🔥 FIRE | State Fire & Emergency Services | Water tenders, aerial ladder (55m), BA vans |
| 🏚️ BUILDING COLLAPSE | USAR Strike Team | Heavy rescue vehicle, K9 squad, 50-ton crane |
| 🚗 ROAD ACCIDENT | Highway Patrol + 108 EMS | Extrication tender (Jaws of Life), trauma ambulances |
| ☣️ GAS LEAK | CBRN Hazmat Taskforce | Level-A hazmat vehicle, decon bus, respiratory units |
| 🏥 MEDICAL | Emergency Medical Services | ALS ambulances, mobile ICU |

---

## 🧪 Preset Emergency Scenarios

| File | Scenario | Language |
|------|----------|----------|
| `patna_flood_hi.wav` | 7 people trapped in urban flood, Patna | Hindi |
| `chennai_fire_ta.wav` | Hospital fire, 6 trapped, Anna Nagar | Tamil |
| `vijayawada_accident_te.wav` | NH-16 bus-truck crash, 8 injured | Telugu |
| `bangalore_collapse_kn.wav` | Construction collapse, 5 trapped, Electronic City | Kannada |
| `howrah_gas_leak_bn.wav` | Chemical factory explosion, 6 trapped | Bengali |
| `delhi_commercial_collapse_en.wav` | Sector 18 complex collapse, 5 trapped | English |

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12, FastAPI, Uvicorn |
| Audio Processing | NumPy, SciPy, SoundFile, PyDub |
| Deep Learning | PyTorch (GRU neural denoiser) |
| ASR | Gnani.ai API + SpeechRecognition |
| Translation | deep-translator (Google Translate) |
| LLM | Google Gemini / Groq LLaMA / Offline NLP |
| Data Validation | Pydantic v2 |
| Frontend | Vanilla HTML5, CSS3, JavaScript (no framework) |
| Audio Synthesis | gTTS + Acoustic formant synthesis |

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
```

### Install Dependencies
```bash
pip install fastapi uvicorn numpy scipy soundfile pydub torch \
            speechrecognition deep-translator gtts google-generativeai \
            groq pydantic python-multipart
```

### Run the Application
```bash
cd disaster-in/backend
python app.py
```

Open your browser at **http://127.0.0.1:8000**

---

## 🔑 API Keys

| Key | Where to get | Required? |
|-----|-------------|-----------|
| Gnani.ai API key | [gnani.ai](https://gnani.ai) | Pre-configured |
| Gemini API key | [ai.google.dev](https://ai.google.dev) | Optional |
| Groq API key | [console.groq.com](https://console.groq.com) | Optional |

Set optional keys as environment variables:
```bash
set GEMINI_API_KEY=your_key_here
set GROQ_API_KEY=your_key_here
```

> The system works **fully offline** without any LLM key — it uses a built-in emergency NLP engine.

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Server status check |
| `GET` | `/api/scenarios` | List preset emergency scenarios |
| `POST` | `/api/process-audio` | Run full 7-stage pipeline |
| `POST` | `/api/operator/confirm` | Operator dispatch authorization |
| `GET` | `/api/incidents` | Incident audit log |
| `GET` | `/api/audio/{category}/{id}` | Stream audio files |

---

## 🧪 Run Integration Test

```bash
cd disaster-in
python test_pipeline.py
```

Tests all 7 stages end-to-end and prints results for each stage.

---

## 📄 License

This project is part of **DisasAI** — an open research initiative for AI-assisted disaster response.

---

<div align="center">
  <b>Built for India's emergency responders 🇮🇳</b><br/>
  Saving lives through AI × Human collaboration
</div>
