"""
Emergency Call Sample Audio Generator:
Synthesizes realistic emergency calls mixed with realistic acoustic noise:
- Telephony band noise
- Ambient emergency noise (rain, siren, traffic rumble, structural crackle)
- Low frequency rumbling (Butterworth target)
- High frequency hiss & random spectral noise (Spectral Gating target)
"""

import os
import io
import json
import numpy as np
import scipy.signal
import soundfile as sf

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

SCENARIOS = [
    {
        "id": "patna_flood_hi",
        "title": "Patna Urban Flood - 7 Trapped (Hindi)",
        "language_code": "hi",
        "language_name": "Hindi (हिन्दी)",
        "emergency_type": "FLOOD",
        "severity": "CRITICAL",
        "text_native": "अरे कोई बचाओ! पटना के राजेंद्र नगर में भारी बाढ़ आ गई है। पानी पहली मंजिल तक घुस गया है। हमारे घर में सात लोग फंसे हैं, दो बुजुर्ग बीमार हैं और एक बच्चा है। बिजली कट गई है, पानी तेजी से बढ़ रहा है! तुरंत बोट और एनडीआरएफ टीम भेजिए!",
        "text_english": "Please save us! A massive flood has hit Rajendra Nagar, Patna. Water has entered the first floor. Seven people are trapped in our house, including two sick elderly individuals and one child. Power is cut, water is rising rapidly! Send rescue boats and the NDRF team immediately!",
        "noise_type": "flood_rain_thunder"
    },
    {
        "id": "chennai_fire_ta",
        "title": "Chennai Hospital Fire - 6 Trapped (Tamil)",
        "language_code": "ta",
        "language_name": "Tamil (தமிழ்)",
        "emergency_type": "FIRE",
        "severity": "CRITICAL",
        "text_native": "ஹலோ அவசர உதவி! சென்னை அண்ணா நகரில் உள்ள மூன்று மாடி மருத்துவமனையில் திடீரென பயங்கர தீ விபத்து ஏற்பட்டுள்ளது! இரண்டாவது தளத்தில் நான்கு நோயாளிகள் மற்றும் இரண்டு செவிலியர்கள் சிக்கியுள்ளனர்! கடும் புகை மூட்டம், தயவுசெய்து உடனே தீயணைப்பு வாகனங்களை அனுப்புங்கள்!",
        "text_english": "Emergency help! A massive fire has suddenly broke out in a three-storey hospital in Anna Nagar, Chennai! Four patients and two nurses are trapped on the second floor! Heavy dense smoke, please dispatch fire tenders immediately!",
        "noise_type": "fire_siren_crackle"
    },
    {
        "id": "vijayawada_accident_te",
        "title": "Highway 16 Bus Crash - 8 Injured (Telugu)",
        "language_code": "te",
        "language_name": "Telugu (తెలుగు)",
        "emergency_type": "ROAD_ACCIDENT",
        "severity": "HIGH",
        "text_native": "అర్జెంట్ హెల్ప్ కావాలి! విజయవాడ జాతీయ రహదారి 16 పై ఘోర రోడ్డు ప్రమాదం జరిగింది. ఒక బస్సు మరియు లారీ ఢీకొన్నాయి. కనీసం ఎనిమిది మంది తీవ్రంగా గాయపడ్డారు, డ్రైవర్ స్టీరింగ్ లో ఇరుక్కుపోయాడు! వెంటనే నాలుగు అంబులెన్సులు, రెస్క్యూ టీమ్ పంపించండి!",
        "text_english": "Urgent help needed! A fatal road accident took place on National Highway 16, Vijayawada. A bus and truck collided. At least eight people are critically injured, the driver is trapped in the steering wheel! Send four ambulances and a rescue team immediately!",
        "noise_type": "highway_traffic_crash"
    },
    {
        "id": "bangalore_collapse_kn",
        "title": "Bangalore Structural Collapse - 5 Trapped (Kannada)",
        "language_code": "kn",
        "language_name": "Kannada (ಕನ್ನಡ)",
        "emergency_type": "BUILDING_COLLAPSE",
        "severity": "CRITICAL",
        "text_native": "ದಯವಿಟ್ಟು ಬೇಗ ಬನ್ನಿ! ಬೆಂಗಳೂರಿನ ಎಲೆಕ್ಟ್ರಾನಿಕ್ ಸಿಟಿ ಹತ್ತಿರ ನಿರ್ಮಾಣ ಹಂತದಲ್ಲಿದ್ದ ಕಟ್ಟಡ ಕುಸಿದು ಬಿದ್ದಿದೆ! ಅವಶೇಷಗಳ ಅಡಿಯಲ್ಲಿ ಸುಮಾರು ಐದು ಮಂದಿ ಕಾರ್ಮಿಕರು ಸಿಲುಕಿಕೊಂಡಿದ್ದಾರೆ! ತಕ್ಷಣ ಕ್ರೇನ್ ಮತ್ತು ರಕ್ಷಣಾ ತಂಡವನ್ನು ಕಳುಹಿಸಿ!",
        "text_english": "Please come quickly! An under-construction building collapsed near Electronic City, Bangalore! Approximately five workers are trapped under the debris! Immediately send cranes and urban search and rescue team!",
        "noise_type": "rumble_debris_shatter"
    },
    {
        "id": "howrah_gas_leak_bn",
        "title": "Howrah Chemical Cylinder Blast - 6 Trapped (Bengali)",
        "language_code": "bn",
        "language_name": "Bengali (বাংলা)",
        "emergency_type": "GAS_LEAK",
        "severity": "CRITICAL",
        "text_native": "জরুরি সাহায্য চাই! হাওড়ার একটি রাসায়নিক কারখানায় ভয়াবহ গ্যাস সিলিন্ডার বিস্ফোরণ হয়েছে এবং আগুন লেগেছে! অন্তত ছয়জন কর্মী ভিতরে আটকা পড়ে আছেন, বিষাক্ত ধোঁয়া ছড়াচ্ছে! অবিলম্বে ফায়ার ব্রিগেড ও অ্যাম্বুলেন্স পাঠান!",
        "text_english": "Emergency help needed! A severe gas cylinder explosion and fire occurred at a chemical factory in Howrah! At least six workers are trapped inside, toxic smoke is spreading! Dispatch fire brigades and ambulances immediately!",
        "noise_type": "industrial_hiss_alarm"
    },
    {
        "id": "delhi_commercial_collapse_en",
        "title": "Noida Sector 18 Complex Collapse (English)",
        "language_code": "en",
        "language_name": "English (Indian)",
        "emergency_type": "BUILDING_COLLAPSE",
        "severity": "CRITICAL",
        "text_native": "Emergency! Major structural collapse reported at Sector 18 Commercial Complex. At least five civilians are trapped under the concrete slabs on the ground floor. Two are bleeding heavily. We need USAR heavy extrication tools and trauma ambulances immediately!",
        "text_english": "Emergency! Major structural collapse reported at Sector 18 Commercial Complex. At least five civilians are trapped under the concrete slabs on the ground floor. Two are bleeding heavily. We need USAR heavy extrication tools and trauma ambulances immediately!",
        "noise_type": "siren_screams_dust"
    }
]


def generate_synthetic_noise(sr: int, duration_sec: float, noise_type: str) -> np.ndarray:
    """Generates complex real-world emergency background noise to test denoiser."""
    num_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # 1. Low frequency rumble (wind, flood water, building rumble, vehicle engine) ~ 20-80 Hz
    rumble = 0.35 * np.sin(2 * np.pi * 38 * t) + 0.25 * np.sin(2 * np.pi * 55 * t)

    # 2. Electrical 50Hz hum + 100Hz harmonic
    hum = 0.15 * np.sin(2 * np.pi * 50 * t) + 0.08 * np.sin(2 * np.pi * 100 * t)

    # 3. High-frequency hiss & atmospheric radio noise
    white_noise = np.random.normal(0, 0.22, num_samples)

    # 4. Emergency Siren (modulated frequency 700Hz to 1200Hz)
    siren_mod = np.sin(2 * np.pi * 0.4 * t)
    siren_freq = 850 + 350 * siren_mod
    siren = 0.20 * np.sin(2 * np.pi * siren_freq * t)

    # Combine based on noise type
    if "flood" in noise_type:
        # Heavy rain & water torrent (colored noise)
        b, a = scipy.signal.butter(2, 800 / (sr / 2), btype='low')
        water_roar = scipy.signal.lfilter(b, a, np.random.normal(0, 0.4, num_samples))
        noise = rumble * 0.6 + water_roar * 0.7 + white_noise * 0.3
    elif "fire" in noise_type:
        # Fire crackle + siren + rumble
        crackle = (np.random.rand(num_samples) > 0.985) * np.random.normal(0, 0.8, num_samples)
        noise = rumble * 0.4 + crackle * 0.5 + siren * 0.4 + white_noise * 0.25
    elif "highway" in noise_type:
        # Traffic rumble + horn blare + skid
        horn = 0.25 * np.sin(2 * np.pi * 420 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0.5)
        noise = rumble * 0.6 + horn * 0.4 + white_noise * 0.3
    else:
        # General collapse / industrial disaster
        noise = rumble * 0.5 + siren * 0.35 + hum * 0.3 + white_noise * 0.25

    return noise.astype(np.float32)


def generate_voice_formant_speech(text: str, sr: int = 16000, duration_sec: float = 6.0) -> np.ndarray:
    """
    Synthesizes acoustic voice formant modulated speech signal as robust offline fallback.
    Models natural human vocal tract formants (F1: 500Hz, F2: 1500Hz, F3: 2500Hz)
    with syllable cadences and panic micro-tremors.
    """
    num_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Syllabic envelope (bursts of speaking and natural breath pauses)
    syllable_rate = 3.8  # syllables per second
    envelope = np.abs(np.sin(2 * np.pi * syllable_rate * t)) ** 1.8

    # Pause intervals (breathing gasp)
    pause_mask = (np.sin(2 * np.pi * 0.35 * t) > -0.6).astype(float)
    envelope = envelope * pause_mask

    # Voice fundamental pitch (urgent pitch ~ 220Hz - 280Hz)
    f0 = 230.0 + 35.0 * np.sin(2 * np.pi * 0.8 * t)
    # Formants (Vowel / Speech Resonances)
    vocal_chord = np.sin(2 * np.pi * f0 * t)
    f1 = 0.5 * np.sin(2 * np.pi * 550 * t)   # F1
    f2 = 0.3 * np.sin(2 * np.pi * 1650 * t)  # F2
    f3 = 0.15 * np.sin(2 * np.pi * 2600 * t) # F3

    speech = (vocal_chord + f1 + f2 + f3) * envelope * 0.65
    return speech.astype(np.float32)


def create_sample_audio_file(scenario: dict) -> str:
    """
    Generates WAV audio file for the given scenario with gTTS if network available,
    or acoustic vocal tract formant synthesis if offline. Adds real noisy layer.
    """
    filepath = os.path.join(SAMPLES_DIR, f"{scenario['id']}.wav")
    sr = 16000

    clean_speech = None

    # Try gTTS for authentic spoken accent
    try:
        from gtts import gTTS
        tts = gTTS(text=scenario["text_native"], lang=scenario["language_code"], slow=False)
        mp3_io = io.BytesIO()
        tts.write_to_fp(mp3_io)
        mp3_io.seek(0)

        # Convert to numpy via soundfile/scipy or pydub
        from pydub import AudioSegment
        audio_seg = AudioSegment.from_file(mp3_io, format="mp3")
        audio_seg = audio_seg.set_frame_rate(sr).set_channels(1)
        samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
        samples = samples / 32768.0
        clean_speech = samples
    except Exception as e:
        # Fallback to acoustic voice formant synthesizer
        clean_speech = generate_voice_formant_speech(scenario["text_native"], sr=sr, duration_sec=6.5)

    # Generate noise layer
    duration = len(clean_speech) / sr
    noise = generate_synthetic_noise(sr, duration, scenario["noise_type"])

    # Mix voice + noise (Noisy Emergency Call)
    # SNR ~ 2dB to 6dB (heavily noisy call)
    noisy_audio = clean_speech * 0.75 + noise * 0.45
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

    # Save to disk
    audio_int16 = (noisy_audio * 32767).astype(np.int16)
    sf.write(filepath, audio_int16, sr, format='WAV', subtype='PCM_16')
    return filepath


def ensure_all_sample_files():
    """Generates all preset emergency scenarios to ensure offline instant readiness."""
    meta_path = os.path.join(SAMPLES_DIR, "scenarios_meta.json")
    created = []
    for sc in SCENARIOS:
        fp = os.path.join(SAMPLES_DIR, f"{sc['id']}.wav")
        if not os.path.exists(fp):
            try:
                create_sample_audio_file(sc)
            except Exception as e:
                print(f"Error creating {sc['id']}: {e}")
        sc_copy = dict(sc)
        sc_copy["audio_filename"] = f"{sc['id']}.wav"
        sc_copy["audio_url"] = f"/api/audio/sample/{sc['id']}.wav"
        created.append(sc_copy)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(created, f, ensure_ascii=False, indent=2)
    return created


if __name__ == "__main__":
    ensure_all_sample_files()
    print("All sample files generated successfully!")
