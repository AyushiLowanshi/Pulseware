# Pulseware â€” Heart Rhythm & Health Prediction

Pulseware is a decision-support prototype that helps you understand your heart rhythm from everyday wearable data. It learns what is normal for *you* from your history and flags new readings that look different from your usual pattern. It is not a medical device.

### Features

**Tier 1: Continuous Monitoring**
- Builds a personalized baseline from your PPG-derived history (BPM, SDNN, RMSSD)
- Scores new readings as Normal, Monitor, or Flag for review with an explanation
- Handles post-exercise recovery (2-hour window â†’ Expected recovery) and supports symptom-based escalation
- Upload a reading via CSV, Excel, image or PDF â€” screenshots are read with Google Gemini Vision AI

**History**
- View your recent heart rate & HRV chart and all recent checks in one place

**Tier 2: ECG Snapshot**
- On-demand classification of single-lead ECG snapshots (Normal, PVC-like, AFib-like, Bradycardia-like, Abnormal)
- Upload an ECG as image, PDF or CSV, or enter values manually
- Safety guardrails check heart rate, QTc, QRS width and P-wave
- Trained on MIT-BIH Arrhythmia Database when available, otherwise synthetic data

### Tech Stack

- **Streamlit** (Python) â€” `app.py`, `anomaly_model.py`, `ecg_classifier.py`
- **React + Vite** â€” `web-app/src/App.jsx`
- **Models:** Isolation Forest (personal baseline), Random Forest (ECG)
- **Vision:** Google Gemini for screenshot/PDF reading

### Project Structure

```
Pulseware/
â”œâ”€â”€ app.py                    # Streamlit app
â”œâ”€â”€ .streamlit/config.toml    # Theme
â”œâ”€â”€ anomaly_model.py          # Tier 1 baseline model
â”œâ”€â”€ ecg_classifier.py         # Tier 2 ECG classifier
â”œâ”€â”€ field_aliases.py          # Field definitions
â”œâ”€â”€ file_reading.py           # File + Gemini reading
â”œâ”€â”€ baseline_threshold.py
â”œâ”€â”€ generate_dummy_data.py    # Sample data
â”œâ”€â”€ ppg_preprocessing.py      # PPG filtering & features
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ screenshots/
â””â”€â”€ web-app/                  # React app
    â”œâ”€â”€ src/App.jsx
    â”œâ”€â”€ src/styles.css
    â”œâ”€â”€ src/main.jsx
    â”œâ”€â”€ index.html
    â”œâ”€â”€ vite.config.js
    â””â”€â”€ package.json
```

### How to Run

**Streamlit**

```bash
pip install -r requirements.txt
streamlit run app.py
# open http://localhost:8501
```

**React**

```bash
cd web-app
npm install
cp .env.example .env   # add VITE_GEMINI_API_KEY if you want screenshot reading
npm run dev            # open http://localhost:5173
npm run build          # for production
```

The Gemini API key is optional. Get a free key at https://aistudio.google.com and set it as `GEMINI_API_KEY` (Streamlit) or `VITE_GEMINI_API_KEY` (React). Without it, you can still use CSV/manual entry.

### Authors

Yash Jashnani, Ayushi Lowanshi, Latifa Zaman

### Disclaimer

Pulseware is a research prototype for educational purposes. It does not provide medical diagnosis. Always consult a doctor for health concerns.
