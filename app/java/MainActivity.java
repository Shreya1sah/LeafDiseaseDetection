package com.example.fuzzy_ensemble_technique_wala_app;
//package com.example.plantdiseaseclassifier;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import android.os.Bundle;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    private static final int IMAGE_PICK_CODE = 1000;

    private TFLiteModel denseNetModel, inceptionModel, xceptionModel;
    private final int INPUT_SIZE = 128;//size ka gadbadi hai jaise 256 ya 224 dene se error de raha tha
    private final int NUM_CLASSES = 5;

    private final String[] CLASS_NAMES = {
            "Apple_Scab", "Black_Rot", "Cedar_Apple_Rust", "Healthy", "not_leaf"
    };

    private ImageView imageView;
    private TextView resultText;
    private Button predictButton, selectButton;

    private Bitmap selectedBitmap = null;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        resultText = findViewById(R.id.result_text);
        predictButton = findViewById(R.id.predict_button);
        selectButton = findViewById(R.id.select_image_button);

        predictButton.setEnabled(false); // Disabled until image is selected

        // Load the models
        try {
            denseNetModel = new TFLiteModel(getAssets(), "densenet_model.tflite", INPUT_SIZE, NUM_CLASSES);
            inceptionModel = new TFLiteModel(getAssets(), "inception_model.tflite", INPUT_SIZE, NUM_CLASSES);
            xceptionModel = new TFLiteModel(getAssets(), "xception_model.tflite", INPUT_SIZE, NUM_CLASSES);
        } catch (IOException e) {
            e.printStackTrace();
            resultText.setText("Error loading models.");
        }

        // Select Image from gallery
        selectButton.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(intent, IMAGE_PICK_CODE);
        });

        // Predict button
        predictButton.setOnClickListener(v -> {
            if (selectedBitmap != null) {
                classifyImage(selectedBitmap);
            } else {
                resultText.setText("Please select an image first.");
            }
        });

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }
    // Handle selected image from gallery
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == IMAGE_PICK_CODE && resultCode == RESULT_OK && data != null) {
            try {
                InputStream inputStream = getContentResolver().openInputStream(data.getData());
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                selectedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
                imageView.setImageBitmap(selectedBitmap);
                predictButton.setEnabled(true);
                resultText.setText("Image ready for prediction.");
            } catch (Exception e) {
                e.printStackTrace();
                resultText.setText("Failed to load image.");
            }
        }
    }

    // Classify the selected image using ensemble model
    private void classifyImage(Bitmap bitmap) {
        try {
            float[] pred1 = denseNetModel.predict(bitmap);
            float[] pred2 = inceptionModel.predict(bitmap);
            float[] pred3 = xceptionModel.predict(bitmap);

            int finalPrediction = EnsembleFuzzyFusion.fuzzyRankBasedFusion(pred1, pred2, pred3);
            String result = "Predicted Class: " + CLASS_NAMES[finalPrediction];

            resultText.setText(result);
        } catch (Exception e) {
            e.printStackTrace();
            resultText.setText("Prediction failed: " + e.getMessage());
        }
    }

}