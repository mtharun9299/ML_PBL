After running the Python training pipeline, copy:
1. ml_pipeline/models/crop_disease_model_quantized.tflite
2. ml_pipeline/models/class_names.json

into this assets folder.

Copy:
3. ml_pipeline/treatment_data.json

into this assets folder.

The binary TFLite model is intentionally not bundled because it is generated from the actual dataset during training.
