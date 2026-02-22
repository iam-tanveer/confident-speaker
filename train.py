import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- 1. CONFIGURATION ---
DATA_DIR = "data" # Your folder containing 'high', 'medium', 'low' folders
CLASSES = {"low": 0, "high": 1}

def extract_features(file_path):
    """Extracts the mean MFCCs from an audio file, handling variable lengths."""
    try:
        # Load audio (librosa handles .mp3 automatically if ffmpeg is installed)
        y, sr = librosa.load(file_path, sr=22050)
        # Extract MFCCs (40 bands is standard for speech)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # Take the mean across the time axis
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

# --- 2. LOAD DATA ---
print("Extracting features from 3800 files... (Grab a coffee, this takes a few minutes)")
X = []
y_labels = []

for label_name, label_idx in CLASSES.items():
    folder_path = os.path.join(DATA_DIR, label_name)
    
    # 1. Check if folder exists
    if not os.path.exists(folder_path):
        print(f"⚠️ WARNING: Could not find folder: '{folder_path}'")
        continue
        
    # 2. Count files loaded per folder
    files_loaded = 0
    for filename in os.listdir(folder_path):
        if filename.endswith((".wav", ".mp3")):
            file_path = os.path.join(folder_path, filename)
            features = extract_features(file_path)
            
            if features is not None:
                X.append(features)
                y_labels.append(label_idx)
                files_loaded += 1
                
    print(f"✅ Loaded {files_loaded} files from '{label_name}'")

X = np.array(X)
y_labels = np.array(y_labels)

# 3. Failsafe: Check how many classes were actually loaded
unique_classes = np.unique(y_labels)
print(f"\nFound {len(X)} total files across {len(unique_classes)} classes.")

if len(unique_classes) < 2:
    raise ValueError("❌ ERROR: The model needs at least 2 classes to train! Please check your folder names and ensure they contain .mp3 files.")

# --- 3. TRAIN THE MODEL ---
print("Splitting data and training the model...")
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)

# Random Forest is highly robust against overfitting for tabular feature data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. EVALUATE & SAVE ---
predictions = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions, target_names=CLASSES.keys()))

# Save the trained model to disk so Streamlit can use it
joblib.dump(model, "confidence_model.pkl")
print("Model saved successfully as 'confidence_model.pkl'!")