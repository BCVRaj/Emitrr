# Medical Transcript Analysis Pipeline

Production-ready medical conversation analysis with hybrid AI architecture.

## Overview

This pipeline analyzes medical transcripts to extract:
- **Medical Entities** - symptoms, diseases, treatments, anatomy, medications
- **Medical Summary** - chief complaint, diagnoses, prognosis, treatment plans
- **Sentiment & Intent** - patient emotional state and communication goals
- **SOAP Notes** - structured clinical documentation

## Architecture

**Hybrid AI System:**
- **NER**: `d4data/biomedical-ner-all` (local, ~300MB)
- **Sentiment**: `distilbert-base-uncased-finetuned-sst-2-english` (local, ~255MB)
- **Intent & Summaries**: Google Gemini API (cloud)
- **Design**: Schema-based, config-driven, modular

## 📁 Project Structure

```
Emitrr/
│
├── data/
│   ├── input/                          # Put transcript files here
│   │   └── transcript.txt
│   └── output/                         # Generated results (timestamped)
│       ├── YYYYMMDD_HHMMSS_medical_summary.json
│       ├── YYYYMMDD_HHMMSS_sentiment_intent.json
│       ├── YYYYMMDD_HHMMSS_soap_note.json
│       ├── YYYYMMDD_HHMMSS_entities.json
│       └── YYYYMMDD_HHMMSS_complete_results.json
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                # Clean text, split speakers
│   ├── ner_extractor.py               # Local medical NER extraction
│   └── llm_extractor.py               # Hybrid: DistilBERT + Gemini
│
├── app.py                              # Streamlit web interface
├── schemas.py                          # JSON field definitions
├── config.py                           # All settings & patterns
├── main.py                             # CLI pipeline orchestrator
├── requirements.txt
├── .env                                # API key (DON'T commit!)
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Two Ways to Use:

**Option 1: Web Interface (Recommended)** 🌐
- User-friendly GUI
- Upload files via browser
- Visual results display
- Download buttons

**Option 2: Command Line** 💻
- Fast for automation
- Script integration
- Batch processing

---

### 1. Clone/Download Project

```bash
cd Emitrr
```

### 2. Install Dependencies

# Install packages
pip install -r requirements.txt
```

**Note**: First run downloads models:
- NER model: ~300MB
- Sentiment model: ~255MB
- Total: ~555MB, requires ~2GB RAM

### 3. Setup API Key

Get your Gemini API key from: https://makersuite.google.com/app/apikey

Edit `.env`:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Prepare Transcript

Create `data/input/transcript.txt`:

```
Doctor: Good morning! How can I help you today?
Patient: I've been having severe headaches for the past 3 days.
Doctor: Can you describe the pain?
Patient: It's a throbbing pain on the left side of my head.
Doctor: Any nausea or sensitivity to light?
Patient: Yes, both. And I feel dizzy sometimes.
Doctor: Based on your symptoms, this appears to be a migraine. I'll prescribe sumatriptan 50mg.
Patient: Will this help with the nausea too?
Doctor: Yes, it should. Take it at the onset of symptoms. Let's schedule a follow-up in 2 weeks.
Patient: Thank you, doctor. I feel better knowing what it is.
```

### 5. Run Pipeline

**Option A: Web Interface (Streamlit)**

```bash
streamlit run app.py
```

Opens in browser at `http://localhost:8501`

**Features:**
- 📁 Upload transcript files via drag-and-drop
- 👀 Preview transcript before analysis
- 🚀 One-click analysis
- 📊 Interactive results display with tabs
- 💾 Download JSON outputs
- 🎨 Clean, professional UI

**Option B: Command Line**

```bash
python main.py data/input/transcript.txt
```

Outputs saved to `data/output/` with timestamps.

---

## 🌐 Streamlit Web Interface

### Features:

**1. File Upload**
- Drag-and-drop or browse for `.txt` files
- Instant file preview
- Character count display

**2. Analysis Dashboard**
- Real-time progress indicators
- Step-by-step processing feedback
- Error handling with clear messages

**3. Results Display**
- **Medical Summary Tab**: Patient name, symptoms, diagnosis, treatment, prognosis
- **Sentiment & Intent Tab**: Emotional state analysis (DistilBERT) + Intent detection (Gemini)
- **Entities Tab**: Categorized medical entities with confidence scores, interactive table
- **SOAP Note Tab**: Expandable sections (Subjective, Objective, Assessment, Plan)
- **Downloads Tab**: Individual JSON downloads or complete results

**4. User Experience**
- Clean, minimal interface
- No technical knowledge required
- Instant visual feedback
- Professional presentation

### Running the Web Interface:

```bash
# Start the server
streamlit run app.py

# Access in browser
# http://localhost:8501
```

The interface automatically uses the same backend pipeline as the CLI, ensuring consistent results.

---

## 📊 Output Files

All outputs saved to `data/output/` with timestamps:

### 1. **medical_summary.json**
```json
{
  "chief_complaint": "Severe headaches for 3 days",
  "symptoms": ["headaches", "throbbing pain", "nausea", "dizziness"],
  "diagnoses": ["migraine"],
  "treatments": ["sumatriptan 50mg"],
  "timeline": "Past 3 days",
  "notes": "Pain localized to left side of head"
}
```

### 2. **sentiment_intent.json**
```json
{
  "Sentiment": "Reassured",
  "Intent": "Seeking reassurance and treatment"
}
```
- **Sentiment**: DistilBERT model (local, Transformer-based)
- **Intent**: Gemini API (cloud, contextual understanding)

### 3. **soap_note.json**
```json
{
  "subjective": "Patient reports severe headaches for 3 days...",
  "objective": "Throbbing pain localized to left side...",
  "assessment": "Clinical presentation consistent with migraine...",
  "plan": "Prescribe sumatriptan 50mg, follow-up in 2 weeks..."
}
```

### 4. **entities.json**
```json
{
  "all_entities": [
    {"text": "headaches", "type": "SYMPTOM", "confidence": 0.95},
    {"text": "migraine", "type": "DISEASE", "confidence": 0.92}
  ],
  "categorized": {
    "symptoms": [...],
    "diseases": [...],
    "treatments": [...],
    "medications": [...]
  },
  "statistics": {
    "total": 15,
    "average_confidence": 0.89
  }
}
```

### 5. **complete_results.json**
- All outputs combined in one file

## ⚙️ Configuration

### Adjust Settings in `config.py`:

```python
CONFIG = {
    "ner": {
        "confidence_threshold": 0.8,  # Higher = fewer entities
    },
    "sentiment": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "positive_threshold": 0.65,  # Sentiment mapping
        "negative_threshold": 0.65,
    },
    "gemini": {
        "model_name": "gemini-1.5-flash",  # Or "gemini-1.5-pro"
        "temperature": 0.0,  # 0 = deterministic, 1 = creative
    },
    "text_limits": {
        "ner_max_chars": 5000,
        "llm_max_chars": 4000,
    },
    "entity_categories": {
        "symptoms": ["SYMPTOM", "SIGN", "FINDING"],
        # Add custom categories here
    }
}
```

### Add Speaker Patterns in `config.py`:

```python
"speakers": {
    "patterns": {
        "doctor": [
            r'(doctor|physician|dr\.?)\s*:',
            r'MD\s*:',
            # Add your patterns
        ],
        "patient": [
            r'(patient|pt\.?)\s*:',
            # Add your patterns
        ]
    }
}
```

## 🧪 Testing

```bash
# Test with sample transcript
python main.py data/input/transcript.txt

# Check outputs
ls data/output/

# View logs
cat pipeline.log
```

## 🔧 Troubleshooting

### Issue: "GEMINI_API_KEY not found"
**Solution**: Make sure `.env` file exists with valid API key

### Issue: NER model download fails
**Solution**: Check internet connection, try again (model caches locally)

### Issue: "No doctor/patient utterances found"
**Solution**: Check transcript format matches speaker patterns in `config.py`

### Issue: Low entity confidence
**Solution**: Reduce `confidence_threshold` in `config.py` (e.g., 0.6)

### Issue: JSON parsing errors from Gemini
**Solution**: Increase `temperature` slightly or switch to `gemini-1.5-pro`

## Performance

- **NER Extraction**: ~2-5 seconds (local, first run downloads model)
- **Sentiment Analysis**: ~0.5-2 seconds (local, DistilBERT)
- **Gemini API Calls**: ~3-10 seconds (depends on network)
- **Total Pipeline**: ~10-20 seconds per transcript

**Note**: First run downloads models (~555MB total), subsequent runs use cached models.

## Security

- API key stored in `.env` (never committed to git)
- All processing configurable
- Detailed logging for audit trails
- Hybrid architecture: sentiment runs locally, only intent/summaries use cloud API

## Features

### Professional Architecture
- Schema-based: single source of truth for field names
- Config-driven: no hardcoded values
- Modular: clean separation of concerns
- Hybrid AI: local models + cloud LLM for optimal performance
- **Dual Interface**: Web UI (Streamlit) + Command Line

### Web Interface (Streamlit)
- User-friendly GUI
- Drag-and-drop file upload
- Real-time progress tracking
- Interactive results display
- One-click JSON downloads
- No technical knowledge required

### Entity Extraction
- Local biomedical NER model
- Configurable confidence thresholds
- Automatic categorization
- Deduplication
- True confidence scores

### Sentiment Analysis
- DistilBERT Transformer model (local)
- Task specification compliant
- Fast inference
- Accurate emotion detection
- Mapped to medical context: Reassured/Neutral/Anxious

### Intent Detection
- Gemini API (contextual understanding)
- Medical conversation aware
- High accuracy for complex intents

### LLM Analysis
- Structured JSON outputs
- Schema validation
- Fallback handling
- Deterministic results

### SOAP Note Generation
- Medical standard format
- Context-aware
- Entity-informed

## 🛠️ Customization

### Add New Entity Category

In `config.py`:
```python
"entity_categories": {
    "lab_tests": ["LAB", "TEST", "MEASUREMENT"],
    # Your new category
}
```

### Change Output Format

In `schemas.py`:
```python
class MedicalSummaryFields:
    NEW_FIELD = "new_field"
    
    @classmethod
    def get_default(cls):
        return {
            cls.NEW_FIELD: "default_value"
        }
```

### Switch to Better LLM Model

In `config.py`:
```python
"gemini": {
    "model_name": "gemini-1.5-pro",  # More accurate
}
```

## Dependencies

- `transformers>=4.38.0` - Hugging Face transformers (NER + Sentiment)
- `torch>=2.0.0` - PyTorch backend
- `google-generativeai>=0.3.0` - Gemini API
- `python-dotenv>=1.0.0` - Environment variables
- `numpy>=1.24.0` - Numerical operations
- `pandas>=2.0.0` - Data manipulation
- `streamlit>=1.31.0` - Web interface

## Requirements

- **Python**: 3.8+
- **RAM**: 2GB minimum (for models)
- **Storage**: ~1GB (models + cache)
- **Internet**: Required for Gemini API and first-time model downloads

