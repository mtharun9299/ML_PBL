# Exact run guide

## A. Train the ML model

1. Install Python.
2. Open a terminal in the project folder.
3. Install dependencies:
   `pip install -r ml_pipeline/requirements.txt`
4. Download/prepare PlantVillage.
5. Put class folders under:
   `ml_pipeline/data/plantvillage/`
6. Run:
   `python ml_pipeline/train.py --data ml_pipeline/data/plantvillage`

For the first run, use:
`python ml_pipeline/train.py --data ml_pipeline/data/plantvillage --baseline-epochs 1 --finetune-epochs 1`

This checks that TensorFlow, the dataset, model creation and TFLite conversion all work.

For the final training, use the default 15 + 10 epochs.

## B. Validate the trained model

Run:
`python ml_pipeline/test_tflite.py --model ml_pipeline/models/crop_disease_model_quantized.tflite --image YOUR_IMAGE.jpg --classes ml_pipeline/models/class_names.json`

Run Grad-CAM:
`python ml_pipeline/gradcam.py --model ml_pipeline/models/crop_disease_model.keras --image YOUR_IMAGE.jpg --classes ml_pipeline/models/class_names.json`

## C. Build Android app

Open `android_app` in Android Studio.

Copy:
- `crop_disease_model_quantized.tflite`
- `class_names.json`
- `treatment_data.json`

to:
`android_app/app/src/main/assets/`

Then Build > Make Project and run on a physical Android phone.

## D. Offline verification

After installation:
1. Enable airplane mode.
2. Open the application.
3. Select a leaf image from Gallery.
4. Verify prediction and treatment text appear.
5. Camera can also be tested if camera permission is granted.

## E. Troubleshooting

### "Model setup failed"
The TFLite model is missing from assets. Train first and copy it into the assets directory.

### Wrong number of classes
The app uses `class_names.json`, so the order must exactly match the output order produced by the training script.

### Low accuracy
Use the full dataset and default training schedule. Check class imbalance, image quality and the confusion matrix.

### Android Gradle dependency issue
Open the project with a recent Android Studio and allow Gradle to sync. Internet is required for the first Gradle dependency download; inference itself is offline.
