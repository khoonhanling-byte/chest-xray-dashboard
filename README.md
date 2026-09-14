# Chest X-ray AI Dashboard — Visual V3

FYP2 Streamlit dashboard for Normal vs Pneumonia chest X-ray classification using VGG16, DenseNet121 and ResNet50.

## Main user features

- Original vs CLAHE-enhanced X-ray
- Recommended model or manual model selection
- Normal-like / Pneumonia-like result
- Model confidence and threshold proximity warning
- Grad-CAM visual explanation
- Downloadable analysis summary
- Three-model comparison with model agreement and approximate inference time

## Research features

- Labelled Test Model page using exported test-set samples
- False-negative and false-positive gallery
- ROC and Precision–Recall curves
- Per-model confusion matrices
- Training accuracy/loss curves
- Dataset class distribution
- Session analysis history

The earlier standalone dense-unit and freeze-ratio effect panels were removed from the Model Performance page. The selected settings can still appear in the benchmark table for reproducibility.

## Install the visual update

If you already have a working dashboard with the three `.keras` files:

1. Stop Streamlit with `Ctrl + C`.
2. Back up your current project.
3. Replace the project code with this V3 folder, but keep your existing `models/` folder.
4. Restart:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The basic prediction pages work with your existing model files.

## Install the new research-support data

The Test Model, Misclassified Cases, ROC/PR curves, confusion matrices and training-curve sections require the new Colab export.

Run:

`FYP2_Three_Models_30plus30_DashboardV3.ipynb`

At the end it creates:

`three_models_for_dashboard_v3.zip`

Install it from the dashboard folder:

```powershell
.\.venv\Scripts\python.exe install_models.py "C:\path\to\three_models_for_dashboard_v3.zip"
```

Then restart Streamlit.

## Expected model files

```text
models/
├── vgg16_best.keras
├── densenet121_best.keras
├── resnet50_best.keras
├── best_models_summary.json
└── class_names.json
```

## Important

This is an educational research prototype and is not a medical diagnosis system.
