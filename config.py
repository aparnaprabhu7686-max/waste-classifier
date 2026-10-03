"""Central configuration for the Smart Waste Classification System."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"          # dataset/<class_name>/*.jpg
MODEL_PATH = BASE_DIR / "model" / "waste_model.keras"
LABELS_PATH = BASE_DIR / "model" / "labels.json"

CLASSES = ["glass", "metal", "organic", "paper", "plastic"]  # alphabetical = Keras order
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 8        # training with frozen MobileNetV2 base
EPOCHS_FINE = 6        # fine-tuning top layers
SEED = 42

# Disposal tips shown in the UI
TIPS = {
    "plastic": {"bin": "Dry waste (Blue bin)", "tip": "Rinse and dry the item. Recyclable plastics go to the recycling centre."},
    "paper":   {"bin": "Dry waste (Blue bin)", "tip": "Keep paper dry and clean. Greasy or wet paper is not recyclable."},
    "metal":   {"bin": "Dry waste (Blue bin)", "tip": "Rinse cans and tins. Scrap metal can be sold to recyclers."},
    "glass":   {"bin": "Dry waste (Special glass bin)", "tip": "Handle carefully. Do not mix broken glass with other waste."},
    "organic": {"bin": "Wet waste (Green bin)", "tip": "Compost it! Food and garden waste make good manure."},
}
