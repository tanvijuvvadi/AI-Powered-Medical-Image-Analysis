# MedVision AI — AI-Powered Medical Image Analysis

A responsive, GitHub Pages-ready educational website for a student computer-vision project exploring chest X-ray classification workflows and responsible AI.

> **Safety notice:** This repository currently contains a front-end prototype only. The image result is randomly simulated; it is not produced by a trained model and must never be used for diagnosis, triage, or treatment decisions.

## Features
- Responsive research dashboard and project overview
- Local image preview for JPG, PNG, and WEBP files
- Clearly labeled simulated output panel
- AI pipeline explainer from public data through evaluation
- Evaluation metric definitions and model validation checklist
- Dataset, privacy, fairness, and limitation notes
- No framework or build step required

## Run locally
1. Download or clone this repository.
2. Open `index.html` in a modern browser, or run a simple local server:
   ```bash
   python -m http.server 8000
   ```
3. Visit `http://localhost:8000`.

## Project files
```text
AI-Medical-Image-Analysis/
├── index.html      # Website structure and content
├── styles.css      # Responsive styling
├── script.js       # Local preview and explicitly simulated demo
├── README.md       # Project documentation
├── .gitignore      # Files excluded from Git
└── docs/
    └── MODEL_INTEGRATION.md
```

## Planned model integration
A real research implementation could use a public, appropriately licensed chest X-ray dataset, TensorFlow/Keras CNN, patient-aware data splits, a held-out test set, and metrics such as sensitivity, specificity, precision, recall, F1, ROC-AUC, and a confusion matrix. Do not display performance values until the model has actually been trained and evaluated. See [`docs/MODEL_INTEGRATION.md`](docs/MODEL_INTEGRATION.md).

## Dataset reference
Kermany et al. pediatric chest X-ray dataset: https://data.mendeley.com/datasets/rscbjbr9sj/2. Review the dataset documentation and terms before use. Dataset composition and labels impose important limits on generalization.

## Publish with GitHub Pages
1. Create a public repository named `ai-medical-image-analysis`.
2. Add `index.html`, `styles.css`, `script.js`, `README.md`, `.gitignore`, and `docs/`.
3. Open **Settings → Pages**.
4. Under build and deployment, choose **Deploy from a branch**, select `main` and `/ (root)`, then save.
5. Wait for GitHub Pages to publish the site. The Pages URL appears in Settings → Pages.

## Suggested Git commands
```bash
git init
git add .
git commit -m "Build responsive MedVision AI project website"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai-medical-image-analysis.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. If the repository already exists locally, do not run `git init` again.

## Responsible-use principles
- Educational research only; not a medical device.
- Never upload identifiable patient data to an unapproved service.
- Do not interpret this demo's random label or confidence as clinical evidence.
- A future real model needs robust validation, privacy/security controls, bias analysis, clinical oversight, and applicable regulatory review.

## License
Choose a license only after confirming the licensing of any data, model weights, and third-party assets you add. The original starter website code may be licensed under MIT if you choose to include an MIT license.
