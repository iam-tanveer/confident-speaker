# 🎤 Confident Speaker

**Vocal Coaching powered by AI.**
*Built for HackED 2026*

---

## 📖 The Vision

Public speaking is scary (we are experiencing it first-hand before our presentations), but for many who struggle with social cues, or people who aren't native to English, understanding exactly how their vocal tone is being perceived by others can be difficult, and it can also be embarrassing to ask for feedback from others.

## ✨ Features

* **Real-Time Acoustic Feedback:** Get instant analysis on volume and pacing (WPM) using `librosa` and Google `SpeechRecognition`.
* **Accessible UI (Universal Design):** High-contrast, clean layout designed specifically to reduce cognitive load, adhering to DivE Accessibility standards.
* **Persuasive Gamification:** Integrated streak tracking, XP systems, and positive reinforcement loops to encourage consistent practice.

## ⚠️ Important: Domain Shift & Hardware Calibration

Machine learning audio models are highly susceptible to domain shift. The base architecture was trained on 3,000 cleanly cropped dataset files. However, live laptop microphones introduce heavy compression, baseline static, and variable room acoustics.

> **Note:** The pre-trained `confidence_model.pkl` included in this repository is currently calibrated to a specific presenter's hardware for the live HackED 2026 demo.
If you clone this repository and experience inaccurate confidence scores, the model is simply interpreting your microphone's baseline compression as "nervous hesitation." To fix this, the dataset needs to be updated and model needs to be finetuned using that.

## 💻 Tech Stack

* **Frontend:** Streamlit (Python)
* **AI/ML:** scikit-learn (Random Forest Classifier)
* **Audio Processing:** `librosa`
* **Feature Extraction:** Mean MFCCs (Mel-Frequency Cepstral Coefficients)
