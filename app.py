import streamlit as st
import librosa
import numpy as np
import speech_recognition as sr
import time
import tempfile
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="VocaConfidence", page_icon="🎤", layout="centered")

# --- LOAD ML MODEL (CACHED) ---
#loads into memory once, preventing app lag
@st.cache_resource
def load_ml_model():
    try:
        return joblib.load("confidence_model.pkl")
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        return None

trained_model = load_ml_model()

# --- INITIALIZE GAMIFICATION STATE ---
# This acts as our temporary database to track the user's progress and streaks
if 'streak' not in st.session_state:
    st.session_state.streak = 1
if 'xp' not in st.session_state:
    st.session_state.xp = 0
if 'sessions_completed' not in st.session_state:
    st.session_state.sessions_completed = 0

# --- AI INFERENCE ---
def get_confidence_score(audio_path):
    if trained_model is None:
        return 0 
        
    try:
        # 1. Load the live audio
        y, sr_rate = librosa.load(audio_path, sr=22050)
        
        # 3. Extract MFCCs from the trimmed audio, not the raw audio
        mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        features_reshaped = mfccs_mean.reshape(1, -1)
        
        # Predict probabilities: [Low(0), High(1)]
        probabilities = trained_model.predict_proba(features_reshaped)[0]
        prob_low, prob_high = probabilities
        print(prob_low, prob_high)
        
        # Convert probabilities into a Confidence Score out of 100
        score = (prob_high * 100) + (prob_low * 0)
        
        # Give the score a slight boost to account for laptop mic quality
        adjusted_score = min(100, int(score * 1.1)) 
        
        return adjusted_score
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0

# --- ACOUSTIC METRICS ANALYSIS ---
def analyze_audio(audio_path):
    y, sr_rate = librosa.load(audio_path, sr=None)
    
    # Loudness
    rms = librosa.feature.rms(y=y)
    loudness = np.mean(librosa.amplitude_to_db(rms, ref=np.max)) + 100
    
    # Pitch
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=2000)
    valid_f0 = f0[~np.isnan(f0)]
    pitch = np.mean(valid_f0) if len(valid_f0) > 0 else 0
    
    # Clarity (WPM via SpeechRecognition)
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            word_count = len(text.split())
            duration_minutes = librosa.get_duration(y=y, sr=sr_rate) / 60.0
            wpm = int(word_count / duration_minutes) if duration_minutes > 0 else 0
        except sr.UnknownValueError:
            text = "Could not transcribe audio clearly."
            wpm = 0
            
    return pitch, loudness, wpm, text

# --- FRONTEND APP LAYOUT ---
st.title("🎤 VocaConfidence")
st.markdown("### Your trusted public speaking coach")

# --- NAVIGATION TABS ---
tab_practice, tab_profile = st.tabs(["🎯 Practice", "👤 My Profile"])

with tab_practice:
    practice_type = st.selectbox(
        "Choose Your Practice", 
        ["One-on-One Interview", "Group Presentation", "Quick Pitch"]
    )
    
    st.markdown("---")
    audio_value = st.audio_input("Record your practice session")

    if audio_value is not None:
        # Save live audio to a temporary file for librosa & SpeechRecognition
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_value.read())
            tmp_file_path = tmp_file.name

        with st.spinner("Analyzing vocal confidence..."):
            time.sleep(1.5) # Slight delay for dramatic effect
            pitch, loudness, wpm, transcript = analyze_audio(tmp_file_path)
            confidence_score = get_confidence_score(tmp_file_path)
            
            # --- GAMIFICATION LOGIC ---
            st.session_state.sessions_completed += 1
            xp_gained = max(10, confidence_score * 2) # Guarantee some XP
            st.session_state.xp += int(xp_gained)
            
            st.balloons()

        
        
        st.success(f"Great Speech! You earned +{int(xp_gained)} XP! 🏆")
        st.metric(label="Overall Confidence Score", value=f"{confidence_score}%")
        
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Average Pitch", value=f"{int(pitch)} Hz")
        col2.metric(label="Loudness", value=f"{int(loudness)} dB")
        col3.metric(label="Clarity (Pace)", value=f"{wpm} WPM")
            
        st.markdown("### Personalized Coaching Tips")
        if loudness < 65:
            st.warning("🗣️ **Volume:** Try projecting your voice from your diaphragm.")
        elif loudness > 85:
            st.warning("🗣️ **Volume:** You are speaking quite loudly, try a calmer delivery.")
        else:
            st.info("🗣️ **Volume:** Perfect volume range.")
            
        if wpm > 160:
            st.warning("⏱️ **Pace:** You are speaking very fast, take a breath and slow down.")
        elif 0 < wpm < 110:
            st.warning("⏱️ **Pace:** You are speaking a bit slowly, try picking up the pace.")
            
        with st.expander("View Speech Transcript"):
            st.write(transcript)

with tab_profile:
    st.header("Your Progress")
    st.markdown("Keep up the great work! Consistent practice builds real confidence.")
    
    # --- SELF-MONITORING DASHBOARD ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Streak", f"{st.session_state.streak} Days 🔥")
    col2.metric("Total XP", f"{st.session_state.xp} 🌟")
    col3.metric("Sessions", f"{st.session_state.sessions_completed} 🎤")
    
    st.markdown("### Achievements")
    if st.session_state.sessions_completed >= 1:
        st.success("✅ **First Steps:** Completed first practice session!")
    else:
        st.info("🔒 **First Steps:** Complete your first practice session to unlock.")
        
    if st.session_state.xp >= 1000:
        st.success("✅ **Confidence Master:** Earned 1,000 XP!")
    else:
        st.info(f"🔒 **Confidence Master:** Earn {1000 - st.session_state.xp} more XP to unlock.")