package com.example.cropdiseasedetector;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.HashMap;

import org.json.JSONArray;
import org.json.JSONObject;

public class MainActivity extends AppCompatActivity {

    private static final int IMAGE_SIZE = 224;
    private ImageView leafImage;
    private TextView resultText;
    private TextView treatmentText;
    private ProgressBar progress;

    private Interpreter interpreter;
    private List<String> classNames = new ArrayList<>();
    private Map<String, String> treatmentMap = new HashMap<>();
    private Bitmap currentBitmap;

    private final ActivityResultLauncher<String> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    try {
                        Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                        useImage(bitmap);
                    } catch (Exception e) {
                        showError("Could not open image: " + e.getMessage());
                    }
                }
            });

    private final ActivityResultLauncher<Intent> cameraLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                    Bundle extras = result.getData().getExtras();
                    if (extras != null && extras.get("data") instanceof Bitmap) {
                        useImage((Bitmap) extras.get("data"));
                    }
                }
            });

    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) openCamera();
                else showError("Camera permission is required for camera capture.");
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        leafImage = findViewById(R.id.leafImage);
        resultText = findViewById(R.id.resultText);
        treatmentText = findViewById(R.id.treatmentText);
        progress = findViewById(R.id.progress);

        findViewById(R.id.galleryButton).setOnClickListener(v -> galleryLauncher.launch("image/*"));
        findViewById(R.id.cameraButton).setOnClickListener(v -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
            }
        });

        try {
            interpreter = new Interpreter(loadModelFile("crop_disease_model_quantized.tflite"));
            loadClasses();
            loadTreatments();
        } catch (Exception e) {
            showError("Model setup failed. Copy the generated TFLite model into app/src/main/assets/. " + e.getMessage());
        }
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        cameraLauncher.launch(intent);
    }

    private void useImage(Bitmap bitmap) {
        currentBitmap = bitmap;
        leafImage.setImageBitmap(bitmap);
        runInference(bitmap);
    }

    private void runInference(Bitmap source) {
        if (interpreter == null) {
            showError("TFLite model is not loaded.");
            return;
        }

        progress.setVisibility(View.VISIBLE);
        resultText.setText("Running offline inference...");
        treatmentText.setText("");

        new Thread(() -> {
            try {
                Bitmap resized = Bitmap.createScaledBitmap(source, IMAGE_SIZE, IMAGE_SIZE, true);
                Tensor inputTensor = interpreter.getInputTensor(0);
                DataType inputType = inputTensor.dataType();

                ByteBuffer input = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * 3);
                input.order(ByteOrder.nativeOrder());

                float inputScale = inputTensor.quantizationParams().getScale();
                int inputZero = inputTensor.quantizationParams().getZeroPoint();

                int[] pixels = new int[IMAGE_SIZE * IMAGE_SIZE];
                resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

                for (int pixel : pixels) {
                    float r = ((pixel >> 16) & 0xFF) / 255.0f;
                    float g = ((pixel >> 8) & 0xFF) / 255.0f;
                    float b = (pixel & 0xFF) / 255.0f;

                    if (inputType == DataType.INT8) {
                        input.put((byte) Math.max(-128, Math.min(127,
                                Math.round(r / inputScale + inputZero))));
                        input.put((byte) Math.max(-128, Math.min(127,
                                Math.round(g / inputScale + inputZero))));
                        input.put((byte) Math.max(-128, Math.min(127,
                                Math.round(b / inputScale + inputZero))));
                    } else {
                        input.put((byte) Math.round(r * 255));
                        input.put((byte) Math.round(g * 255));
                        input.put((byte) Math.round(b * 255));
                    }
                }

                Tensor outputTensor = interpreter.getOutputTensor(0);
                int outputClasses = outputTensor.shape()[outputTensor.shape().length - 1];
                ByteBuffer output = ByteBuffer.allocateDirect(outputClasses);
                output.order(ByteOrder.nativeOrder());

                interpreter.run(input, output);
                output.rewind();

                float outScale = outputTensor.quantizationParams().getScale();
                int outZero = outputTensor.quantizationParams().getZeroPoint();

                float[] probs = new float[outputClasses];
                for (int i = 0; i < outputClasses; i++) {
                    int raw = output.get();
                    float value = (raw - outZero) * outScale;
                    probs[i] = Math.max(0f, value);
                }

                float sum = 0f;
                for (float p : probs) sum += p;
                if (sum > 0f) {
                    for (int i = 0; i < probs.length; i++) probs[i] /= sum;
                }

                int best = 0;
                for (int i = 1; i < probs.length; i++) {
                    if (probs[i] > probs[best]) best = i;
                }

                final int index = best;
                final float confidence = probs[best];

                runOnUiThread(() -> {
                    progress.setVisibility(View.GONE);
                    String className = index < classNames.size()
                            ? classNames.get(index) : "Class " + index;
                    String displayName = prettyName(className);

                    resultText.setText(String.format(Locale.US,
                            "Prediction: %s\nConfidence: %.2f%%",
                            displayName, confidence * 100f));

                    String treatment = treatmentMap.get(className);
                    if (treatment == null) {
                        treatment = "No local treatment entry is available for this class. Consult a qualified agricultural expert.";
                    }
                    treatmentText.setText("Recommended action:\n" + treatment);
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    progress.setVisibility(View.GONE);
                    showError("Inference failed: " + e.getMessage());
                });
            }
        }).start();
    }

    private String prettyName(String raw) {
        return raw.replace("___", " - ")
                .replace("_", " ")
                .replace("(", "")
                .replace(")", "")
                .trim();
    }

    private ByteBuffer loadModelFile(String filename) throws IOException {
        try (InputStream is = getAssets().open(filename)) {
            byte[] bytes = new byte[is.available()];
            int read = is.read(bytes);
            if (read != bytes.length) throw new IOException("Could not read complete model file.");
            ByteBuffer buffer = ByteBuffer.allocateDirect(bytes.length);
            buffer.order(ByteOrder.nativeOrder());
            buffer.put(bytes);
            buffer.rewind();
            return buffer;
        }
    }

    private void loadClasses() throws Exception {
        String json = readAsset("class_names.json");
        JSONArray arr = new JSONArray(json);
        classNames.clear();
        for (int i = 0; i < arr.length(); i++) classNames.add(arr.getString(i));
    }

    private void loadTreatments() throws Exception {
        String json = readAsset("treatment_data.json");
        JSONObject obj = new JSONObject(json);
        Iterator<String> keys = obj.keys();
        while (keys.hasNext()) {
            String key = keys.next();
            JSONObject item = obj.getJSONObject(key);
            treatmentMap.put(key, item.optString("treatment", ""));
        }
    }

    private String readAsset(String name) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(getAssets().open(name)))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line);
        }
        return sb.toString();
    }

    private void showError(String message) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
        resultText.setText(message);
    }

    @Override
    protected void onDestroy() {
        if (interpreter != null) interpreter.close();
        super.onDestroy();
    }
}
