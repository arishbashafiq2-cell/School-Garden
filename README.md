# School Garden Watering and Leaf Image Assistant

AI for Everyone – Project 04. A classroom prototype for a teacher or student volunteer caring for a demonstration school garden.

**Live app:** _paste your Streamlit Community Cloud link here_
**Team:** _names and roles_

## What the app does (five tabs)

| Tab | What it shows | Model |
|---|---|---|
| 1 · Garden overview and watering review | Data overview, then a watering review flag for one record (example, typed, or CSV) | Decision tree (`models/supervised.joblib`) |
| 2 · Condition profiles | The record's observation group and the three group profiles | StandardScaler + KMeans, k = 3 (`models/clusters.joblib`) |
| 3 · Bean leaf image | Dataset class for a sample or uploaded bean leaf, with a human check | Small CNN (`models/images.keras`) |
| 4 · Garden comments | Sentiment and keyword topics of the 30 supplied comments | NLTK VADER + `data/topic_keywords.json` |
| 5 · Volunteer summary | **Offline summary** (default, Python template) and an optional Groq Free Plan summary | Groq `openai/gpt-oss-20b` (optional) |

The app only loads saved models. It never retrains after a click.

## Data (supplied – nothing was collected)
- `data/main.csv` – 1000 **generated** classroom scenarios with fixed splits (700 train / 150 validation / 150 test). `needs_water` = 1 when simulated next-morning moisture falls below 25% – a teaching threshold, not advice for real plants.
- `data/feedback.csv` – 30 comments authored for teaching, with reference sentiment labels.
- Bean leaf images – public Beans dataset (Makerere AI Lab). The full training arrays (`images.npz`) are **not** in this repository; only trained models and three sample images.
- See `source_notices/` for attribution and licences.

## Repository layout
```
app.py                 Streamlit interface and validation functions
requirements.txt       Runtime packages, pinned to the training versions
models/                supervised.joblib, clusters.joblib, images.keras,
                       cluster_descriptions.json, metadata.json
data/                  main.csv, feedback.csv, example_inputs.csv, meta.json,
                       topic_keywords.json, prompt_test_cases.json
nltk_data/             Offline VADER lexicon (no download at startup)
sample_images/         One example image per class
source_notices/        Attribution and licences
.streamlit/secrets.toml.example   Template for the optional Groq key
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Use the same Python minor version as Colab (shown at the end of the training notebook).

## Deploy (free)
1. **GitHub Free:** New repository → *Add file → Upload files* → drag in everything above (not `secrets.toml`, not `images.npz`, not ZIPs) → *Commit changes*. Note: the web uploader takes files up to 25 MB, so the model files fit.
2. **Streamlit Community Cloud:** *Create app* → choose the repository, branch `main`, file `app.py`. Under **Advanced settings** pick the Python version printed by the training notebook. Deploy.
3. **Optional Groq key:** in the app's *Settings → Secrets*, paste:
   ```toml
   GROQ_API_KEY = "your-free-plan-key"
   GROQ_MODEL = "openai/gpt-oss-20b"
   ```
   Without a key the app works fully in Offline summary mode. Keep the Groq account on the Free Plan.

If a free hosting limit blocks the live demo, run the app locally and submit a screen recording.

## Results (fill in from your own Colab run – test split, evaluated once)
| Model | Metric | Result | Baseline |
|---|---|---|---|
| Decision tree (max_depth 4) | Accuracy / macro F1 / class 1 precision / recall | | Most-frequent class |
| CNN (3 epochs) | Accuracy / macro F1 | | – |
| VADER | Accuracy / macro F1 vs supplied labels | | – |

## Acceptance checks (record what you observed)
| Test | Expected result | Observed | Fix |
|---|---|---|---|
| Open the app without a Groq key | All data, prediction, group, image and feedback functions work | | |
| Select a supplied example record | Flag and its review meaning appear | | |
| Use a supplied sample image | Processed 96×96 RGB image, class from saved order | | |
| Submit a blank comment | Treated as empty, not counted | | |
| Enter negative `days_since_water` | Rejected: "must be between 0 and 5 days" | | |
| Upload a CSV missing a column | Clear message, no prediction | | |
| Simulate an API failure (tab 5 checkbox) | Error message, Offline summary still shown | | |
| Restart the app / rerun a widget | Models load, no training runs | | |
| Open the repository on another computer | README and files are enough to run it | | |

## Limitations
- Sensor results come from generated data; their relationships are part of the teaching recipe.
- Leaf labels are dataset classes for bean leaves only – not a diagnosis, and not valid for other plants. Scores are not calibrated probabilities.
- A leaf image does not determine watering need: the tabular and image models answer different questions.
- VADER and keyword topics give tone and subject only; negation, sarcasm and mixed opinions are hard. English only.
- The AI summary may be wrong; every number must be checked against the app's results.
