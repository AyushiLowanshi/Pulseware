# Heart Rhythm & Health Prediction — UI redesign

Both front-ends were restyled to follow **Apple Health's** design language:
a grouped light-gray canvas, white inset cards with hairline separators (no
borders, no shadows), system type with a large bold title and small uppercase
section headers, one category tint (heart pink `#FF2D55`) plus iOS blue for
actions, and semantic green / orange / red used *only* for status.

All features, flows, model logic, prompts, CSV/alias handling and Gemini
reading are unchanged — only presentation and layout moved.

```
heart-app/
├── app.py                    Streamlit app (restyled; logic untouched)
├── .streamlit/config.toml    Streamlit theme (canvas, card, accent colours, font)
├── anomaly_model.py          unchanged
├── ecg_classifier.py         unchanged
├── field_aliases.py          unchanged
├── file_reading.py           unchanged (API key now read from GEMINI_API_KEY first)
├── baseline_threshold.py     unchanged
├── generate_dummy_data.py    sample-data generators the Streamlit app imports
├── requirements.txt
├── screenshots/              what the redesign looks like
└── web-app/                  React app (Vite)
    ├── src/App.jsx           redesigned UI, same state/logic
    ├── src/styles.css        the design system (new — replaces inline styles)
    ├── src/main.jsx          imports styles.css
    ├── index.html, vite.config.js, package.json
    └── .env.example          VITE_GEMINI_API_KEY for screenshot / PDF reading
```

## Run

```bash
# React
cd web-app
npm install
cp .env.example .env        # add your Gemini key if you want image/PDF reading
npm run dev                 # http://localhost:5173

# Streamlit
pip install -r requirements.txt
streamlit run app.py        # http://localhost:8501
```

If you are dropping this into your existing repo, the only React files you
need are `web-app/src/App.jsx`, `web-app/src/styles.css` and the one-line
`import "./styles.css";` in `main.jsx`. For Streamlit, copy `app.py`,
`.streamlit/config.toml` and `generate_dummy_data.py` (if you don't already
have one).

