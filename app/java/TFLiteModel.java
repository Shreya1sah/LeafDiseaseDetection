package com.example.smartleafai;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
public class TFLiteModel {
    private Interpreter interpreter;
    private int inputSize;
    private int numClasses;

    public TFLiteModel(AssetManager assetManager, String modelPath, int inputSize, int numClasses) throws IOException {
        this.inputSize = inputSize;
        this.numClasses = numClasses;
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath));
        // 🟡 Inspect and log the input tensor's shape and data type
        int[] inputShape = interpreter.getInputTensor(0).shape();
        DataType inputType = interpreter.getInputTensor(0).dataType();

        Log.d("ModelInfo", "Input shape: " + Arrays.toString(inputShape));
        Log.d("ModelInfo", "Input data type: " + inputType.name());

    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    //    public float[] predict(Bitmap bitmap) {
//        // Resize the image
//        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);
//
//        // Load into TensorImage using FLOAT32
//        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
//        tensorImage.load(resizedBitmap);
//
//        // Output buffer for results
//        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, numClasses}, DataType.FLOAT32);
//
//        // Run inference
//        interpreter.run(tensorImage.getBuffer(), outputBuffer.getBuffer());
//
//        return outputBuffer.getFloatArray();
//    }
    public float[] predict(Bitmap bitmap) {
        // Resize the image
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);

        // Normalize pixel values to [0, 1]
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[inputSize * inputSize];
        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize);

        for (int i = 0; i < intValues.length; ++i) {
            int val = intValues[i];
            float r = ((val >> 16) & 0xFF) / 255.0f;
            float g = ((val >> 8) & 0xFF) / 255.0f;
            float b = (val & 0xFF) / 255.0f;
            inputBuffer.putFloat(r);
            inputBuffer.putFloat(g);
            inputBuffer.putFloat(b);
        }

        // Output buffer for results
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, numClasses}, DataType.FLOAT32);

        // Run inference
        interpreter.run(inputBuffer, outputBuffer.getBuffer());

        return outputBuffer.getFloatArray();
    }





}
