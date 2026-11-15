import streamlit as st
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Custom Modules
from modules.data_loader import FloodDataIngestor
from modules.flood_models import UNetSusceptibility, ConvLSTMForecast
from modules.risk_assesment import RiskAssessmentEngine
from modules.alert_generator import MultiModeAlertGenerator
from modules.translation_voice import VoiceTranslationEngine

DEFAULT_GNANI_KEY = "vach_1ytE2CY5X2EAx0JSa6lbDstJx56124bUdRAGHZ31eBVRaixPhFwAxg3ZVZ8TsrcZcDmi0BxluzMRgjBsyDONja51wPnwM900_67b138e9cf1a534c085baee34e4e269b"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Custom Satellite & GPS Flood Forecasting",
    page_icon="📡",
    layout="wide"
)

# --- CACHED MODEL LOADERS ---
@st.cache_resource
def load_deep_learning_models():
    unet = UNetSusceptibility(in_channels=4, out_channels=1)
    convlstm = ConvLSTMForecast(input_dim=1, hidden_dim=16)
    unet.eval()
    convlstm.eval()
    return unet, convlstm

st.title("📡 Satellite Ingestion & Geotargeted Emergency Voice Gateway")
st.caption("Upload Satellite Image ──► Enter Lat/Long ──► PyTorch U-Net Prediction ──► Auto-Detect Regional Language ──► Gnani Voice Broadcast")

# --- SIDEBAR: USER INPUT CONTROLS ---
st.sidebar.header("📍 1. Custom Region Coordinates")
location_name = st.sidebar.text_input("Location Name", value="Aluva Sector 4")
user_lat = st.sidebar.number_input("Latitude", value=10.1076, format="%.4f")
user_lon = st.sidebar.number_input("Longitude", value=76.3516, format="%.4f")

st.sidebar.header("🖼️ 2. Satellite Image Input")
uploaded_sat_file = st.sidebar.file_uploader(
    "Upload Satellite Raster (PNG, JPG, TIFF)", 
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

st.sidebar.header("⚙️ 3. Weather Simulation & API")
forecast_hours = st.sidebar.slider("Forecast Lead Window (Hours)", 3, 12, 6)
gnani_api_key = st.sidebar.text_input("Gnani.ai API Key", value=DEFAULT_GNANI_KEY, type="password")

# Initialize System Engines
ingestor = FloodDataIngestor(height=64, width=64)
unet_model, convlstm_model = load_deep_learning_models()
risk_engine = RiskAssessmentEngine()
translator = VoiceTranslationEngine(api_key=gnani_api_key)

# --- INGEST & PROCESS SATELLITE IMAGE ---
with st.spinner("Processing satellite imagery and running U-Net inference..."):
    if uploaded_sat_file:
        file_bytes = uploaded_sat_file.read()
        sat_data, raw_preview_img = ingestor.process_uploaded_image(file_bytes)
    else:
        sat_data = ingestor.generate_synthetic_raster(seed=42)
        raw_preview_img = None

    rain_seq = ingestor.generate_rainfall_sequence(timesteps=forecast_hours, seed=42)
    
    with torch.no_grad():
        sat_tensor = torch.from_numpy(sat_data).unsqueeze(0)
        susceptibility_mask = unet_model(sat_tensor).squeeze().numpy()
        
        rain_tensor = torch.from_numpy(rain_seq).unsqueeze(0)
        rain_forecast_mask = convlstm_model(rain_tensor).squeeze().numpy()

    # Evaluate specific Lat/Long Location Risk
    custom_risk_data = risk_engine.evaluate_custom_location(
        lat=user_lat, 
        lon=user_lon, 
        location_name=location_name, 
        susceptibility_map=susceptibility_mask, 
        rainfall_forecast=rain_forecast_mask
    )

# --- MAIN NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs([
    "🗺️ Custom Prediction & Satellite Vision", 
    "📢 Auto-Geotargeted Gnani Voice Alert", 
    "📡 Multi-Channel Dispatch System"
])

# ==========================================
# TAB 1: SATELLITE PREDICTION & MAPS
# ==========================================
with tab1:
    st.subheader("Regional Ingestion & Risk Evaluation")
    
    # Coordinates & Auto-Detected Language Header Banner
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("Target Location", custom_risk_data["village"])
    col_info2.metric("GPS Coordinates", f"{user_lat:.2f}° N, {user_lon:.2f}° E")
    col_info3.metric("Computed Risk Level", custom_risk_data["risk_level"], delta_color="inverse")
    col_info4.metric("Auto-Detected Language", custom_risk_data["detected_language"])

    st.markdown("---")
    
    col_img1, col_img2, col_img3 = st.columns(3)
    
    with col_img1:
        st.markdown("**1. Uploaded / Ingested Satellite Source**")
        if raw_preview_img:
            st.image(raw_preview_img, use_container_width=True)
        else:
            fig1, ax1 = plt.subplots()
            ax1.imshow(sat_data[0], cmap="terrain")
            st.pyplot(fig1)
            st.caption("Using default satellite sample (Upload an image in sidebar to replace)")
            
    with col_img2:
        st.markdown("**2. U-Net Flood Inundation Susceptibility**")
        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(susceptibility_mask, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, label="Probability")
        st.pyplot(fig2)

    with col_img3:
        st.markdown("**3. ConvLSTM Rainfall Forecast (mm/hr)**")
        fig3, ax3 = plt.subplots()
        im3 = ax3.imshow(rain_forecast_mask, cmap="YlGnBu")
        plt.colorbar(im3, ax=ax3, label="mm/hr")
        st.pyplot(fig3)

# ==========================================
# TAB 2: GEOTARGETED TRANSLATION & GNANI TTS
# ==========================================
with tab2:
    st.subheader("Geotargeted Emergency Translation & Voice Synthesis")
    
    col_lang_left, col_lang_right = st.columns(2)
    with col_lang_left:
        st.info(f"🌐 **GPS Auto-Mapped Regional Language:** {custom_risk_data['detected_language']}")
    with col_lang_right:
        selected_language = st.selectbox(
            "Override Language (Optional):", 
            options=list(translator.LANG_CONFIG.keys()), 
            index=list(translator.LANG_CONFIG.keys()).index(custom_risk_data["detected_language"])
        )

    # Generate English Alert
    english_alert = MultiModeAlertGenerator.generate_template_alert(custom_risk_data)
    translated_alert = translator.translate_text(english_alert, selected_language)

    c_alert1, c_alert2 = st.columns(2)
    
    with c_alert1:
        st.markdown("### 📝 Dynamic English Alert")
        st.info(english_alert)
        
    with c_alert2:
        st.markdown(f"### 🌐 Local Translation ({selected_language})")
        st.success(translated_alert)
        
        st.markdown("### 🎙️ Gnani.ai Voice Synthesizer")
        if st.button("Generate Regional Voice Audio"):
            with st.spinner(f"Synthesizing {selected_language} audio via Gnani Vachana..."):
                audio_fp = translator.text_to_speech_bytes(translated_alert, selected_language)
                if audio_fp.getbuffer().nbytes > 0:
                    st.audio(audio_fp, format="audio/wav")
                else:
                    st.error("Audio generation failed. Check API key or connection.")

# ==========================================
# TAB 3: DISPATCH SYSTEM
# ==========================================
with tab3:
    st.subheader("Geotargeted Alert Broadcast Channels")
    
    st.markdown(f"Broadcast warning for **{custom_risk_data['village']}** ({user_lat}, {user_lon}):")
    ch1, ch2, ch3 = st.columns(3)
    send_ivr = ch1.checkbox("Gnani IVR Voice Call", value=True)
    send_sms = ch2.checkbox("Tower Cell Broadcast SMS", value=True)
    send_wapp = ch3.checkbox("WhatsApp Alert", value=True)
    
    if st.button("🚀 Trigger Broadcast"):
        st.success(f"Emergency Broadcast Dispatched to coordinates ({user_lat}, {user_lon})!")
        if send_ivr:
            st.code(f"[Gnani IVR Call] Synthesized {selected_language} voice stream queued for cell towers near ({user_lat}, {user_lon}).", language="bash")
        if send_sms:
            st.code(f"[Cell Broadcast] SMS emergency alert pushed to active mobile devices in geographic sector.", language="bash")