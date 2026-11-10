"""
Central Configuration — API Keys & Service Settings
"""

# Gnani.ai API Key (Speech-to-Text + Translation)
GNANI_API_KEY = "vach_1ytE2CY5X2EAx0JSa6lbDstJx56124bUdRAGHZ31eBVRaixPhFwAxg3ZVZ8Uelv5vdPfTLLjWpqup56cJVA7ypXdEyCT9y9A_7f36a0e38dfdcc920289a4dace57daa3"
GNANI_TOKEN   = GNANI_API_KEY   # Gnani uses the same token for bearer auth
GNANI_ASR_URL = "https://asr.gnani.ai/api/v1/speech-to-text"

# Optional: set via environment variable to override
import os
GNANI_API_KEY = os.getenv("GNANI_API_KEY", GNANI_API_KEY)
GNANI_TOKEN   = os.getenv("GNANI_TOKEN",   GNANI_TOKEN)
GNANI_ASR_URL = os.getenv("GNANI_BASE_URL", GNANI_ASR_URL)

# LLM Keys (optional — offline NLP engine works without them)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
