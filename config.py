from pathlib import Path

APP_TITLE = "Chest X-ray AI Analysis"
APP_SUBTITLE = "Transfer-learning based Normal vs Pneumonia classification"

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
POSITIVE_CLASS_INDEX = 1
DEFAULT_THRESHOLD = 0.50

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_FILES = {
    "VGG16": MODEL_DIR / "vgg16_best.keras",
    "DenseNet121": MODEL_DIR / "densenet121_best.keras",
    "ResNet50": MODEL_DIR / "resnet50_best.keras",
}

MODEL_SUMMARY_FILE = MODEL_DIR / "best_models_summary.json"
FALLBACK_RECOMMENDED_MODEL = "VGG16"

DISCLAIMER = (
    "This dashboard is an educational research prototype. Its output is not a medical diagnosis "
    "and must not replace assessment by a qualified healthcare professional."
)
