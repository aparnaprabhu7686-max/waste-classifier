# ♻️ AI-Based Smart Waste Classification System

Waste photo eduthaa (camera / upload), AI image classification use panni
**plastic, paper, metal, glass, organic** nu identify pannum. Disposal bin + tip um kaatum.

**Tech:** Python · TensorFlow/Keras (MobileNetV2 transfer learning) · Flask · HTML/JS camera UI

## Project structure
```
waste-classifier/
├── app.py               # Flask web app (camera + upload)
├── train.py             # model training
├── predict.py           # command-line prediction
├── classifier.py        # shared inference code
├── prepare_dataset.py   # converts Kaggle 12-class dataset -> 5 classes
├── config.py            # classes, image size, disposal tips
├── templates/index.html # web UI
├── dataset/<class>/     # put training images here
└── model/               # trained model is saved here
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```
Python 3.9 – 3.12 recommended.

## Step 1 – Dataset
Trained model indha zip-la illa — neenga dataset vachu train pannanum.

**Option A:** Kaggle-la "Garbage Classification" (12 classes) dataset download pannunga, then:
```bash
python prepare_dataset.py --src path/to/garbage_classification --dst dataset
```
**Option B:** Unga own photos ah `dataset/plastic`, `dataset/paper`, `dataset/metal`,
`dataset/glass`, `dataset/organic` folders-la podunga (each class ku 200+ images nalladhu).

## Step 2 – Train
```bash
python train.py
```
CPU-la 10–30 min aagalaam. Output: `model/waste_model.keras` + `model/labels.json`.

## Step 3 – Run
```bash
python app.py            # open http://127.0.0.1:5000
python predict.py test.jpg   # or from command line
```

## Tips for better accuracy
- Simple background-la, waste item center-la irukkura photos use pannunga
- Real-world lighting/angles-la photos add pannunga (dataset bias reduce aagum)
- Accuracy low-na `EPOCHS_*` increase pannunga (config.py) or more images add pannunga

## Possible extensions
Live video detection · Raspberry Pi + servo smart-bin · TFLite conversion for mobile ·
more classes (e-waste, hazardous) · bin-fill dashboard
