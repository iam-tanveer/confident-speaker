import streamlit as st
import librosa
import numpy as np
import speech_recognition as sr
import time
import tempfile
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Confident Speaker", page_icon="🎤", layout="centered")

# --- CUSTOM CSS FOR UI ---
st.markdown("""
<style>
    /* Create modern, rounded cards for metrics */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Make the tabs look like a mobile app navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    /* Style the expander to look cleaner */
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #1c83e1;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ML MODEL (CACHED) ---
@st.cache_resource
def load_ml_model():
    try:
        return joblib.load("confidence_model.pkl")
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}. Please run train.py first.")
        return None

trained_model = load_ml_model()

# --- INITIALIZE GAMIFICATION STATE ---
if 'streak' not in st.session_state:
    st.session_state.streak = 1
if 'xp' not in st.session_state:
    st.session_state.xp = 0
if 'sessions_completed' not in st.session_state:
    st.session_state.sessions_completed = 0

# Calculate Level based on XP (Every 100 XP is a new level)
current_level = (st.session_state.xp // 100) + 1
xp_to_next_level = 100 - (st.session_state.xp % 100)
progress_percentage = (st.session_state.xp % 100) / 100.0

# --- AI INFERENCE ---
def get_confidence_score(audio_path):
    if trained_model is None:
        return 0 
        
    try:
        # Load and trim dead silence for accuracy
        y, sr_rate = librosa.load(audio_path, sr=22050)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        features_reshaped = mfccs_mean.reshape(1, -1)
        
        # Predict probabilities: [Low(0), High(1)]
        probabilities = trained_model.predict_proba(features_reshaped)[0]
        prob_low, prob_high = probabilities
        
        score = int(prob_high * 100)
        return score
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0

# --- ACOUSTIC METRICS ANALYSIS ---
def analyze_audio(audio_path):
    y, sr_rate = librosa.load(audio_path, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # Loudness
    rms = librosa.feature.rms(y=y_trimmed)
    loudness = np.mean(librosa.amplitude_to_db(rms, ref=np.max)) + 100
    
    # Pitch
    f0, _, _ = librosa.pyin(y_trimmed, fmin=65, fmax=2000)
    valid_f0 = f0[~np.isnan(f0)]
    pitch = np.mean(valid_f0) if len(valid_f0) > 0 else 0
    
    # Clarity (WPM)
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            word_count = len(text.split())
            duration_minutes = librosa.get_duration(y=y_trimmed, sr=sr_rate) / 60.0
            wpm = int(word_count / duration_minutes) if duration_minutes > 0 else 0
        except sr.UnknownValueError:
            text = "Could not transcribe audio clearly."
            wpm = 0
            
    return pitch, loudness, wpm, text

# --- FRONTEND APP LAYOUT ---
st.title("🎤 Confident Speaker")
st.markdown("##### Your trusted, accessible public speaking coach.")

tab_practice, tab_profile = st.tabs(["🎯 Practice", "👤 My Profile"])

with tab_practice:
    st.markdown("### Start a Session")
    practice_type = st.selectbox(
        "Choose Your Practice Scenario", 
        ["One-on-One Interview", "Group Presentation", "Quick Pitch"]
    )
    
    audio_value = st.audio_input("Record your practice session")

    if audio_value is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_value.read())
            tmp_file_path = tmp_file.name

        with st.spinner("Analyzing acoustic features..."):
            time.sleep(1.5) 
            pitch, loudness, wpm, transcript = analyze_audio(tmp_file_path)
            confidence_score = get_confidence_score(tmp_file_path)
            
            # --- GAMIFICATION LOGIC ---
            st.session_state.sessions_completed += 1
            xp_gained = max(15, confidence_score // 2) 
            st.session_state.xp += int(xp_gained)
            
            # Re-calculate level after XP gain
            new_level = (st.session_state.xp // 100) + 1
            if new_level > current_level:
                st.balloons()
                st.success(f"🎉 LEVEL UP! You are now Speaker Level {new_level}!")
            else:
                st.success(f"Great Session! You earned +{int(xp_gained)} XP! 🏆")

        # --- DYNAMIC FEEDBACK UI ---
        st.markdown("---")
        st.markdown("### 📊 Session Analysis")
        
        # Highlighted main score
        st.metric(label="Overall Confidence Score", value=f"{confidence_score}%")
        
        # Empathetic feedback for low scores (Persuasive Strategy: Tailoring/Praise)
        if confidence_score < 50:
            st.info("💡 **Coach's Note:** It is completely normal to feel nervous! The AI detected a slight drop in your vocal energy. Before your next try, take a deep breath, drop your shoulders, and speak from your chest.")
        elif confidence_score >= 80:
            st.success("🌟 **Coach's Note:** Incredible execution! Your vocal tone is projecting strong, natural authority.")
        
        # Acoustic breakdown
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Avg Pitch", value=f"{int(pitch)} Hz")
        col2.metric(label="Loudness", value=f"{int(loudness)} dB")
        col3.metric(label="Clarity (Pace)", value=f"{wpm} WPM")
            
        st.markdown("### 🛠️ Actionable Adjustments")
        if loudness < 65:
            st.warning("🗣️ **Volume:** Try projecting your voice slightly more. Imagine speaking to someone at the back of the room.")
        elif loudness > 85:
            st.warning("🗣️ **Volume:** You are speaking quite loudly, which can strain your voice. Try a softer, calmer delivery.")
        else:
            st.info("🗣️ **Volume:** Perfect! Your volume is in the ideal conversational range.")
            
        if wpm > 160:
            st.warning("⏱️ **Pace:** You are speaking very fast. Rushing can signal nervousness to an audience. Take intentional pauses between sentences.")
        elif 0 < wpm < 110:
            st.warning("⏱️ **Pace:** You are speaking a bit slowly, which might lose the audience's attention. Try picking up the energy slightly.")
        elif wpm == 0:
            st.warning("⏱️ **Pace:** Could not detect enough clear words to calculate pace.")
        else:
            st.info("⏱️ **Pace:** Excellent pacing. You sound thoughtful and composed.")
            
        with st.expander("📝 View Speech Transcript (Accessibility Text)"):
            st.write(transcript)

with tab_profile:
    st.header("👤 Your Progress")
    st.markdown("Consistent practice builds real confidence. Here is your growth journey.")
    
    # --- LEVEL PROGRESS BAR ---
    st.markdown(f"**Speaker Level: {current_level}**")
    st.progress(progress_percentage)
    st.caption(f"{xp_to_next_level} XP needed to reach Level {current_level + 1}")
    st.markdown("---")
    
    # --- SELF-MONITORING DASHBOARD ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Streak", f"{st.session_state.streak} Days 🔥")
    col2.metric("Total XP", f"{st.session_state.xp} 🌟")
    col3.metric("Sessions", f"{st.session_state.sessions_completed} 🎤")
    
    st.markdown("### 🎖️ Achievements")
    
    # Display badges based on completed sessions
    if st.session_state.sessions_completed >= 1:
        st.success("✅ **First Steps Badge:** Completed your first practice session!")
    else:
        st.info("🔒 **First Steps Badge:** Complete your first practice session to unlock.")
        
    if st.session_state.sessions_completed >= 5:
        st.success("✅ **Consistent Speaker Badge:** Completed 5 practice sessions!")
    else:
        st.info(f"🔒 **Consistent Speaker Badge:** Complete {5 - st.session_state.sessions_completed} more sessions to unlock.")
        
    if st.session_state.xp >= 500:
        st.success("✅ **Confidence Master Badge:** Earned 500 XP!")
    else:
        st.info(f"🔒 **Confidence Master Badge:** Earn {500 - st.session_state.xp} more XP to unlock.")