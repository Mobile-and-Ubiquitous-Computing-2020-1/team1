package com.example.client;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.client.tflite.Classifier;
import com.example.client.tflite.Classifier.Device;
import com.example.client.tflite.Classifier.Model;

import java.io.IOException;
import java.util.List;


public class MainActivity extends AppCompatActivity {
    /* Tag id for logging */
    protected final String tag = MainActivity.class.getSimpleName();

    private Model model = Model.RESNET;
    private Device device = Device.CPU;
    private int numThreads = -1;
    private Classifier classifier;

    //private Bitmap image;
    private int imageSizeX;
    private int imageSizeY;
    private final int orientation = 0;
    private long lastProcessingTimeMs;

    private ImageView imageView;
    private TextView textView;
    private Button button;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        createClassifier(model, device, numThreads);

        /* Load image */
        //imageView = (ImageView)findViewById(R.id.demo_image);
        //image = ((BitmapDrawable)imageView.getDrawable()).getBitmap();

        /* Load text for displaying result */
        textView = (TextView)findViewById(R.id.view_result);

        /* Button for triggering inference*/
        button = (Button)findViewById(R.id.run_inference);
        button.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View v) {
                runInference();
            }
        });
    }

    /* Inference */
    private void runInference() {
        Log.d(tag, "Inference button triggered");
        /* Reshaping image */
        //Bitmap scaledImage = Bitmap.createScaledBitmap(image, imageSizeX, imageSizeY, false);

        if (classifier != null) {
            final long startTime = SystemClock.uptimeMillis();
            //final List<Classifier.Recognition> results = classifier.recognizeImage(scaledImage, orientation);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            //textView.setText(String.format("Result: %s", results));
        }
    }

    /* Initializing Classifier */
    private void createClassifier(Model model, Device device, int numThreads) {
        if (classifier != null) {
            Log.d(tag, "Closing classifier");
            classifier.close();
            classifier = null;
        }
        try {
            Log.d(tag, "Creating classifier");
            classifier = Classifier.create(this, model, device, numThreads);
        } catch (IOException e) {
            Log.e(tag, "Failed to create classifier.");
        }

        imageSizeX = classifier.getImageSizeX();
        imageSizeY = classifier.getImageSizeY();
    }
}
