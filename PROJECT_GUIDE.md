# Project Guide: AI-Powered Medical Image Analysis (A → M)

## A. Project explanation

**Simple language.** Doctors read X-rays by eye. An AI model can be shown thousands of labelled X-rays until it learns patterns that separate healthy from sick lungs. Then, for a new X-ray, it gives a probability and can highlight the region it looked at. It does not replace the doctor; it works like a second pair of eyes that never gets tired.

**Technical language.** Binary image classification with a convolutional neural network. Convolution layers learn hierarchical features (edges → textures → opacities). We use transfer learning: MobileNetV2 pre-trained on ImageNet supplies general features, a new sigmoid head outputs P(pneumonia), trained with binary cross-entropy and class weights, then fine-tuned at a low learning rate.

**Problems it addresses.** Reading backlogs, delayed critical findings, inter-reader variability, shortage of radiologists in rural areas.

**Why it matters by setting**
- *Hospitals / emergency:* triage – urgent studies move to the top of the worklist.
- *Diagnostic labs & radiology centers:* consistency checks, quality control, throughput.
- *Health-tech companies:* the core of products sold to hospitals (detection, worklist prioritisation, reporting assistance).

**Use cases:** pneumonia / TB / nodules on chest X-ray, tumors on brain MRI, fractures on bone X-ray, stroke bleeds on CT. This repo implements the first as a template; the same pipeline fits the others by swapping the dataset.

**How AI helps:** faster triage, a consistent second opinion that can catch missed findings, and heatmaps that show the doctor *where* to look. Final decisions stay with clinicians.

**Workflow:** image collection → preprocessing (resize, normalise, augment) → feature extraction (CNN backbone) → training → prediction → evaluation (sensitivity, specificity, AUC) → visualisation (curves, confusion matrix, Grad-CAM).

> **Accuracy note on the industry list in your brief:** DeepMind's AlphaFold predicts protein structures, not medical images, and IBM Watson Health was sold off in 2022 (now Merative). Don't claim them as imaging-AI references in interviews. Aidoc, Qure.ai, Annalise.ai, Lunit, Siemens Healthineers, GE HealthCare and Philips are safer examples.

## B. Tech stack options

| | A – Easiest | B – Intermediate | C – Advanced |
|---|---|---|---|
| Libraries | Keras, OpenCV, Matplotlib | + MobileNetV2, scikit-learn, Grad-CAM, Flask | PyTorch/MONAI, nnU-Net, DICOM (pydicom), MLflow, Docker |
| Model | Small custom CNN | Transfer learning (MobileNetV2/ResNet50) | U-Net segmentation / YOLO detection / 3D CNN |
| Dataset | Chest X-ray (2 classes) | Chest X-ray + metrics + explainability | NIH ChestX-ray14 (multi-label), BraTS MRI, LUNA16 CT |
| Difficulty | Low | Medium | High |
| GPU | Not needed (slow) | Optional (Colab free T4) | Needed |
| Output | Accuracy number | Metrics, ROC, heatmaps, API | Localisation masks, DICOM pipeline, deployed service |

## C. Selected approach
**Option B.** It runs on a laptop or free Colab, finishes in under an hour, and gives interview-worthy talking points (transfer learning, imbalance, sensitivity/specificity, Grad-CAM, API). Your original Steps 1–6 become: simple CNN = `--baseline` (comparison), MobileNetV2 = main model. Fixes vs the original snippets: 224×224 RGB for the pretrained backbone, proper validation split, class weights, un-shuffled test set (the original `y_pred` comparison would be misaligned if shuffled), `debug=False` in Flask.

## D. Architecture
```
┌────────────┐  ┌──────────────┐  ┌────────────────┐  ┌───────────────┐
│ X-ray JPEG │→ │ Preprocess   │→ │ Augmentation   │→ │ Normalise     │
│ (dataset)  │  │ RGB, 224x224 │  │ (train only)   │  │ [-1, 1]       │
└────────────┘  └──────────────┘  └────────────────┘  └──────┬────────┘
                                                             ▼
                 ┌────────────────────────────────────────────────┐
                 │ MobileNetV2 backbone (feature extraction)       │
                 └──────────────────────┬─────────────────────────┘
                                        ▼
                   GlobalAvgPool → Dropout(0.3) → Dense(1, sigmoid)
                                        │                │
                                        ▼                ▼
                                P(pneumonia)         Grad-CAM
                                        │                │
                                        ▼                ▼
              Evaluation (confusion matrix, ROC)   Heatmap overlay → Flask API / figures
```
Modules: `data_loader` (pipelines, class weights) · `preprocessing` (OpenCV helpers) · `model` (architectures) · `train` · `evaluate` · `gradcam` · `predict` · `visualize` · `api` · `main` (CLI).

## E. Folder structure
```
AI-Medical-Image-Analysis/
├── data/            dataset lives here (git-ignored) + README with download steps
├── notebooks/       optional Colab/Jupyter exploration (add your own EDA notebook)
├── src/             all importable code (config, data, model, train, evaluate, gradcam, api)
├── models/          saved .keras model (commit only if small, or use GitHub Releases)
├── outputs/         generated plots, logs, metrics.json
├── images/          curated screenshots used by the README
├── docs/            this guide
├── tests/           pytest smoke tests
├── README.md  requirements.txt  .gitignore  main.py
```

## F. Installation
- Python **3.10–3.12** (TensorFlow wheels lag behind the newest Python; check before using 3.13+).
- **Windows:** `python -m venv venv` → `venv\Scripts\activate` → `pip install -r requirements.txt`
- **Mac/Linux:** `python3 -m venv venv` → `source venv/bin/activate` → `pip install -r requirements.txt`
- Verify: `python -c "import tensorflow as tf; print(tf.__version__)"`
- Apple Silicon works with standard `tensorflow` on recent versions. No GPU? Use Google Colab (Runtime → T4 GPU), upload the repo, mount Drive for data.

## G. Code
All code is in `src/` and `main.py`; each file has a docstring explaining its role. Key files: `config.py` (all settings), `data_loader.py` (tf.data + class weights), `model.py` (MobileNetV2 + baseline), `train.py` (2-phase training), `evaluate.py` (metrics), `gradcam.py`, `predict.py`, `api.py`.

## H. Simulation
**How the dataset simulates a hospital:** the labelled X-rays play the role of a PACS archive; `train/` is historical studies, `test/` is "today's new patients" the model has never seen; labels are radiologist-confirmed diagnoses.
**How predictions simulate doctor assistance:** the API receives an image, returns a probability + heatmap; a clinician would review flagged cases first.

Workflow:
1. `python main.py explore` – inspect class balance and samples (proof: images 01–03).
2. `python main.py train` – watch logs and curves (proof: 04 + `training_log_*.csv`).
3. `python main.py evaluate` – compute metrics on unseen test data (proof: 05, 06, 07, `metrics.json`).
4. `python main.py predict <image>` on 1 pneumonia + 1 normal image (proof: Grad-CAM figures).
5. `python main.py serve`, call with `curl`, screenshot the JSON (proof: 09).
6. Discuss the errors (false negatives!) in the README – this is what separates a serious project from a tutorial copy.

## I. Execution and expected results
Commands are listed in the README. Runtime: ~20–60 min on CPU, a few minutes on a Colab GPU. Typical expectation for this dataset with transfer learning: validation accuracy in the 90s, test accuracy lower (often mid-80s to low 90s) with high recall for PNEUMONIA and weaker specificity. Your numbers will vary; report what you actually get. Console output ends with a classification report and `outputs/metrics.json`.

## J. GitHub steps
```bash
git init
git add .gitignore README.md requirements.txt
git commit -m "chore: initial project setup"
git branch -M main
git remote add origin https://github.com/<you>/AI-Medical-Image-Analysis.git
git push -u origin main
```
- **Repo name:** `ai-medical-image-analysis` (or `chest-xray-pneumonia-detection`)
- **Description:** "Chest X-ray pneumonia detection with MobileNetV2 transfer learning, Grad-CAM explainability and a Flask API"
- **Topics:** `deep-learning` `medical-imaging` `computer-vision` `tensorflow` `keras` `transfer-learning` `grad-cam` `flask` `healthcare-ai` `python`
- Upload only curated PNGs to `images/` (copy from `outputs/`); a 30–60 s screen-recording (GIF/MP4 < 25 MB) of the API demo in the README helps a lot. Add a `LICENSE`, pin the repo on your profile.
- Do not push the dataset or large model files without Git LFS.

## K. README
See `/README.md` (complete, with a results table for your own numbers).

## L. Commit plan (7 days)
| Day | Work | Commit message |
|---|---|---|
| 1 Setup | repo, venv, folders, .gitignore, requirements | `chore: project skeleton and environment` |
| 2 Dataset | download, `data/README.md`, explore script, plots 01–02 | `feat: dataset exploration and class distribution` |
| 3 Preprocessing | `preprocessing.py`, `data_loader.py`, plot 03 | `feat: preprocessing and tf.data pipeline` |
| 4 Model | `model.py`, baseline + MobileNetV2, model summary | `feat: MobileNetV2 transfer model and baseline CNN` |
| 5 Training | `train.py`, run training, logs and curves | `feat: two-phase training with class weights` |
| 6 Evaluation | `evaluate.py`, `gradcam.py`, `predict.py`, plots 05–08, metrics | `feat: evaluation, ROC and Grad-CAM explainability` |
| 7 Upload | `api.py`, tests, README results + screenshots, tag `v1.0` | `docs: results, screenshots and limitations; feat: Flask API` |

Tip: commit small and often; spread real commits over several days so the history shows genuine progress.

## M. Proof checklist
| Asset | File name | Where |
|---|---|---|
| Sample X-rays | `02_sample_images.png` | `images/`, README |
| Class distribution | `01_class_distribution.png` | `images/` |
| Preprocessing steps | `03_preprocessing_steps.png` | `images/` |
| Training log | `training_log_phase1.csv`, `_phase2.csv` | `outputs/` (commit) |
| Accuracy/loss graph | `04_training_curves.png` | README |
| Confusion matrix | `05_confusion_matrix.png` | README |
| ROC curve | `06_roc_curve.png` | README |
| Predicted vs actual | `07_prediction_grid.png` | README |
| Grad-CAM example | `08_gradcam_example.png` (copy a `prediction_*.png`) | README |
| API screenshot / GIF | `09_api_demo.png` / `demo.gif` | README |
| Metrics | `metrics.json` | `outputs/` |
| GitHub preview | screenshot of repo page + commit graph | LinkedIn / resume |

Naming: numbered prefix keeps README order; reference with relative paths `![](images/05_confusion_matrix.png)`.

**Resume line:** *Built a chest X-ray pneumonia classifier (MobileNetV2 transfer learning, Grad-CAM, Flask API) on public data; evaluated with sensitivity/specificity/ROC-AUC and documented dataset limitations.* Only include numbers you measured.
