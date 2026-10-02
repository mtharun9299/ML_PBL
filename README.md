# Crop Disease Detector for Low-End Phones

Complete Project-Based Learning implementation based on the uploaded PBL report.

## Project goal

An offline crop-disease classifier using:

- MobileNetV2 transfer learning
- 224x224 RGB input
- Data augmentation
- Class weighting
- Fine-tuning
- TensorFlow Lite INT8 quantization
- Evaluation: accuracy, precision, recall, F1, confusion matrix
- Grad-CAM explainability in Python
- Android offline inference
- Local SQLite treatment lookup

The report specifies a PlantVillage subset covering tomato, potato, corn and apple and approximately 15 classes. The training program below discovers the classes from the dataset folders, so it can also work with a different number of folders without changing the source code. The report's target settings include a 70/15/15 split, batch size 32, 224x224 input, Adam fine-tuning LR 1e-5, baseline 15 epochs and 10 fine-tuning epochs. 

## 1. Requirements

### Training
- Python 3.10 or 3.11 recommended
- TensorFlow 2.15+ / compatible Keras
- NumPy
- OpenCV
- scikit-learn
- matplotlib
- seaborn
- pandas
- Pillow

Install:

```bash
pip install -r ml_pipeline/requirements.txt
```

For GPU training, install a TensorFlow version appropriate for your operating system/GPU.

## 2. Dataset layout

Put the PlantVillage images into:

```text
ml_pipeline/data/plantvillage/
    Apple___Apple_scab/
        image1.jpg
        image2.jpg
    Apple___Black_rot/
        ...
    Corn_(maize)___Common_rust_
        ...
    Potato___Early_blight/
        ...
    Tomato___Early_blight/
        ...
```

Only image files are used.

The script automatically:
1. scans class folders;
2. creates a stratified 70/15/15 train/validation/test split;
3. trains the model;
4. saves the class names;
5. evaluates the test set;
6. exports the TFLite model.

## 3. Train

From the project root:

```bash
python ml_pipeline/train.py --data ml_pipeline/data/plantvillage
```

For a quick test:

```bash
python ml_pipeline/train.py --data ml_pipeline/data/plantvillage --baseline-epochs 1 --finetune-epochs 1
```

The quick command is only for verifying the pipeline, not for project accuracy.

Outputs are written to:

```text
ml_pipeline/models/
ml_pipeline/outputs/
```

Important files:
- `crop_disease_model.keras`
- `crop_disease_model_quantized.tflite`
- `class_names.json`
- `model_metadata.json`
- `training_history.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `accuracy_comparison.png`

## 4. Grad-CAM

After training:

```bash
python ml_pipeline/gradcam.py \
  --model ml_pipeline/models/crop_disease_model.keras \
  --image path/to/test_leaf.jpg
```

The output heatmap is saved to:

```text
ml_pipeline/outputs/gradcam/
```

## 5. Test TFLite

```bash
python ml_pipeline/test_tflite.py \
  --model ml_pipeline/models/crop_disease_model_quantized.tflite \
  --image path/to/test_leaf.jpg
```

## 6. Android application

Open `android_app/` in Android Studio.

Then:
1. Build the project.
2. Copy these generated files into `android_app/app/src/main/assets/`:
   - `crop_disease_model_quantized.tflite`
   - `class_names.json`
3. Put `treatment_data.json` in assets (a starter version is already included).
4. Run on an Android phone.

The app is designed to work without internet after installation.

## 7. Android model input

The application:
- obtains a camera/gallery image;
- center-crops it to a square;
- resizes to 224x224;
- normalizes pixels to the range expected by the trained model;
- runs TFLite inference;
- displays the predicted class and confidence;
- looks up treatment information from local SQLite.

## 8. Important accuracy note

The numerical results in the PBL report (for example 93.6% refined accuracy and 92.4% quantized accuracy) are reported project results. Running this source code on a different dataset split, TensorFlow version, random seed, or dataset copy can produce different numbers. Do not claim the report's accuracy as a newly reproduced result unless you actually run the training and evaluation.

## 9. Suggested final workflow

1. Download the PlantVillage dataset.
2. Select the crop classes required by the report.
3. Place them under `ml_pipeline/data/plantvillage`.
4. Run the training script.
5. Check the generated confusion matrix and metrics.
6. Run Grad-CAM on several correct and incorrect predictions.
7. Copy the TFLite model and class list to Android assets.
8. Build and test in airplane mode.
