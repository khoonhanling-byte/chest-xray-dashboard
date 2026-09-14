# Model folder

After running `FYP2_Three_Models_30plus30.ipynb`, Colab creates:

- `vgg16_best.keras`
- `densenet121_best.keras`
- `resnet50_best.keras`
- `best_models_summary.json`
- `class_names.json`

The easiest installation method is to download `three_models_for_dashboard.zip` from Google Drive and run from the dashboard folder:

```bat
python install_models.py "C:\Users\YOUR_NAME\Downloads\three_models_for_dashboard.zip"
```

The installer copies all three `.keras` files into this folder automatically.
