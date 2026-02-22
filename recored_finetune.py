import os
import time
import shutil
import sounddevice as sd
import soundfile as sf

# --- CONFIGURATION ---
DATA_DIR = "data"
SAMPLE_RATE = 22050
DURATION = 5         # Seconds to record per clip
NUM_CLIPS = 10       # How many unique things you will say per category
MULTIPLIER = 10      # How many copies to make of each recording (Oversampling factor)

def record_and_multiply(label):
    folder_path = os.path.join(DATA_DIR, label)
    os.makedirs(folder_path, exist_ok=True)
    
    print(f"\n--- Recording '{label.upper()}' Confidence ---")
    print(f"We will record {NUM_CLIPS} unique clips, {DURATION} seconds each.")
    print(f"The script will automatically multiply these into {NUM_CLIPS * MULTIPLIER} files.")
    input("Press ENTER when you are ready to start...")
    
    for i in range(NUM_CLIPS):
        print(f"\nClip {i+1}/{NUM_CLIPS} - Get ready...")
        time.sleep(1.5)
        
        print("🔴 RECORDING NOW! Speak!")
        # Record audio from the microphone
        audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait() # Wait until the recording duration is finished
        print("⏹️ Recording stopped.")
        
        # 1. Save the original base file
        timestamp = int(time.time())
        base_filename = f"custom_voice_{label}_{timestamp}.wav"
        base_filepath = os.path.join(folder_path, base_filename)
        sf.write(base_filepath, audio_data, SAMPLE_RATE)
        
        # 2. Multiply (Oversample) the file
        for j in range(1, MULTIPLIER):
            copy_filename = f"custom_voice_{label}_{timestamp}_copy{j}.wav"
            copy_filepath = os.path.join(folder_path, copy_filename)
            shutil.copy2(base_filepath, copy_filepath)
            
    print(f"\n✅ Successfully added {NUM_CLIPS * MULTIPLIER} files to the '{label}' folder!")

if __name__ == "__main__":
    print("🎤 VocaConfidence - Voice Calibration Tool")
    
    # Record the confident clips
    print("\n[STEP 1] Be authoritative, project your voice, and speak clearly.")
    record_and_multiply("high")
    
    # Record the nervous clips
    # print("\n[STEP 2] Speak softer, add hesitation ('um', 'uh'), and sound nervous.")
    # record_and_multiply("low")
    
    print("\n🎉 Calibration data generated! Run train.py to update your model.")