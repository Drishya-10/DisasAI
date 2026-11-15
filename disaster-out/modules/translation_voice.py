import io
import re
import logging
import requests
import streamlit as st

# Try importing gTTS as emergency audio fallback
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

# Try importing official Gnani SDK
try:
    from gnani.tts import GnaniTTSClient
    HAS_GNANI_SDK = True
except ImportError:
    HAS_GNANI_SDK = False

DEFAULT_GNANI_KEY = "vach_1ytE2CY5X2EAx0JSa6lbDstJx56124bUdRAGHZ31eBVRaixPhFwAxg3ZVZ8TsrcZcDmi0BxluzMRgjBsyDONja51wPnwM900_67b138e9cf1a534c085baee34e4e269b"


# --- CACHED TRANSLATION HELPER ---
@st.cache_data(show_spinner=False, ttl=3600)
def _perform_cached_translation(cleaned_text: str, target_lang_code: str) -> str:
    """Caches translation results in memory to prevent rate limiting on UI reruns."""
    # 1. Primary Attempt: Google Translator
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source='auto', target=target_lang_code).translate(cleaned_text)
        if translated and translated.strip() != cleaned_text.strip():
            return translated
    except Exception as e:
        logging.warning(f"Google Translation rate limited/failed: {e}")

    # 2. Secondary Fallback: MyMemory Translator
    try:
        from deep_translator import MyMemoryTranslator
        translated = MyMemoryTranslator(source='auto-detect', target=target_lang_code).translate(cleaned_text)
        if translated and translated.strip() != cleaned_text.strip():
            return translated
    except Exception as e:
        logging.warning(f"MyMemory Translation failed: {e}")

    return ""


class VoiceTranslationEngine:
    """
    Multilingual Translation & TTS engine powered by Gnani.ai (Vachana Speech AI)
    with memory caching and multi-provider translation fallbacks.
    """

    LANG_CONFIG = {
        "Hindi":     {"code": "hi-IN", "voice": "Nalini",   "model": "timbre-v2.5", "gtts": "hi"},
        "Malayalam": {"code": "ml-IN", "voice": "Sita",     "model": "timbre-v2.5", "gtts": "ml"},
        "Tamil":     {"code": "ta-IN", "voice": "Kavya",    "model": "timbre-v2.5", "gtts": "ta"},
        "Telugu":    {"code": "te-IN", "voice": "Ananya",   "model": "timbre-v2.5", "gtts": "te"},
        "Bengali":   {"code": "bn-IN", "voice": "Pooja",    "model": "timbre-v2.5", "gtts": "bn"},
        "Marathi":   {"code": "mr-IN", "voice": "Aarti",    "model": "timbre-v2.5", "gtts": "mr"},
        "Kannada":   {"code": "kn-IN", "voice": "Spandana", "model": "timbre-v2.5", "gtts": "kn"},
        "Gujarati":  {"code": "gu-IN", "voice": "Komal",    "model": "timbre-v2.5", "gtts": "gu"},
        "English":   {"code": "en-IN", "voice": "Pranav",   "model": "timbre-v2.5", "gtts": "en"},
    }

    EMERGENCY_STATIC_TRANSLATIONS = {
        "Malayalam": "അടിയന്തിര പ്രളയ മുന്നറിയിപ്പ്! അടുത്ത 6 മുതൽ 12 മണിക്കൂറിനുള്ളിൽ ശക്തമായ വെള്ളപ്പൊക്കത്തിന് സാധ്യതയുണ്ട്. താഴ്ന്ന പ്രദേശങ്ങളിലുള്ളവർ ഉടൻ ദുരിതാശ്വാസ ക്യാമ്പുകളിലേക്ക് മാറുക.",
        "Hindi": "आपातकालीन बाढ़ चेतावनी! अगले 6 से 12 घंटों के भीतर भारी बाढ़ की संभावना है। निचले इलाकों के लोग तुरंत राहत शिविरों में जाएं।",
        "Tamil": "அவசர வெள்ள அபாய எச்சரிக்கை! அடுத்த 6 முதல் 12 மணி நேரத்திற்குள் வெள்ளப்பெருக்கு ஏற்பட வாய்ப்புள்ளது. தாழ்வான பகுதியில் உள்ள மக்கள் உடனடியாக நிவாரண முகாம்களுக்கு செல்லவும்.",
        "Bengali": "জরুরী বন্যা সতর্কতা! আগামী ৬ থেকে ১২ ঘন্টার মধ্যে ভারী বন্যার সম্ভাবনা রয়েছে। নিচু এলাকার মানুষদের অবিলম্বে ত্রাণ শিবিরে যাওয়ার অনুরোধ করা হচ্ছে।"
    }

    def __init__(self, api_key: str = DEFAULT_GNANI_KEY):
        self.api_key = api_key
        self.gnani_client = None
        if HAS_GNANI_SDK:
            try:
                self.gnani_client = GnaniTTSClient(api_key=self.api_key)
            except Exception as e:
                logging.warning(f"SDK init failed: {e}")

    def translate_text(self, text: str, target_lang_name: str) -> str:
        """Translates text with regex sanitization, Streamlit caching, and multi-backend fallbacks."""
        if target_lang_name not in self.LANG_CONFIG or target_lang_name == "English":
            return text

        # Strip brackets/symbols that disrupt translation parsers
        cleaned_text = re.sub(r'[\[\]]', '', text)
        lang_code = self.LANG_CONFIG[target_lang_name]["gtts"]

        # Call cached function
        result = _perform_cached_translation(cleaned_text, lang_code)
        if result:
            return result

        # Return static offline fallback if both live translation providers fail
        return self.EMERGENCY_STATIC_TRANSLATIONS.get(
            target_lang_name,
            f"[{target_lang_name} Emergency Alert]: {cleaned_text}"
        )

    def text_to_speech_bytes(self, text: str, target_lang_name: str) -> io.BytesIO:
        """Synthesizes text into audio bytes via Gnani.ai with automatic fallback."""
        cfg = self.LANG_CONFIG.get(target_lang_name, self.LANG_CONFIG["Hindi"])
        
        # 1. SDK Synthesis
        if self.gnani_client:
            try:
                audio_bytes = self.gnani_client.synthesize(
                    text=text,
                    voice=cfg["voice"],
                    model=cfg["model"],
                    language=cfg["code"],
                    speed=1.0
                )
                if audio_bytes:
                    fp = io.BytesIO(audio_bytes)
                    fp.seek(0)
                    return fp
            except Exception as e:
                logging.warning(f"Gnani SDK error: {e}")

        # 2. Gnani REST API Call
        url = "https://api.vachana.ai/api/v1/tts/inference"
        headers = {
            "X-API-Key-ID": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voice": cfg["voice"],
            "model": cfg["model"],
            "language": cfg["code"],
            "speed": 1.0,
            "audio_config": {
                "sample_rate": 24000,
                "num_channels": 1,
                "sample_width": 2,
                "encoding": "linear_pcm",
                "container": "wav"
            }
        }

        try:
            res = requests.post(url, json=payload, headers=headers, timeout=8)
            if res.status_code == 200 and len(res.content) > 100:
                fp = io.BytesIO(res.content)
                fp.seek(0)
                return fp
        except Exception as e:
            logging.error(f"Gnani REST error: {e}")

        # 3. Emergency gTTS Fallback
        if HAS_GTTS:
            try:
                tts = gTTS(text=text, lang=cfg["gtts"], slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp
            except Exception as e:
                logging.error(f"gTTS fallback error: {e}")

        return io.BytesIO(b"")