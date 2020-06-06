package com.example.client;

import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.InputType;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.example.client.tflite.Classifier;
import com.example.client.tflite.Classifier.Device;
import com.example.client.tflite.Classifier.Model;
import com.example.client.tflite.HttpTask;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;


public class MainActivity extends AppCompatActivity {
    /* Tag id for logging */
    protected final String tag = MainActivity.class.getSimpleName();

    private static Context context;

    private Model model = Model.FACENET;
    private Device device = Device.CPU;
    private int numThreads = -1;
    private Classifier classifier;

    private Bitmap image;
    private int imageSizeX;
    private int imageSizeY;
    private final int orientation = 0;
    private long lastProcessingTimeMs;

    private ImageView imageView;
    private TextView textView;
    private Button button, go_back, correct, wrong;
    private EditText userInput;
    private AlertDialog.Builder builder;
    private String feedbackInput;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        MainActivity.context = getApplicationContext();
        setContentView(R.layout.activity_main);
        Log.d(tag, "Connected");

        Intent intent = getIntent();
        Integer image_id = intent.getIntExtra("position", R.drawable.demo01);

        createClassifier(model, device, numThreads);

        /* Load image */
        imageView = (ImageView)findViewById(R.id.demo_image);
        imageView.setImageResource(image_id);
        image = ((BitmapDrawable)imageView.getDrawable()).getBitmap();

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

        /* Button for going back to imagelist */
        go_back = (Button)findViewById(R.id.go_back);
        go_back.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });

        /* Interface for user feedback */
        correct = (Button)findViewById(R.id.res_correct);
        wrong = (Button)findViewById(R.id.res_wrong);
        correct.setVisibility(View.GONE);
        wrong.setVisibility(View.GONE);
        correct.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View v) {
                /* TODO: Submit current label with intermediate tensor (feedbackInput) */
                Toast toast = Toast.makeText(getApplicationContext(), String.format("(%s) Success :)", feedbackInput), Toast.LENGTH_SHORT);
                toast.show();
                finish();
            }
        });
        wrong.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View v) {
                builder.show();
            }
        });

        /* Popup dialog for user feedback*/
        builder = new AlertDialog.Builder(this);
        builder.setTitle("Feedback");
        userInput = new EditText(this);
        userInput.setInputType(InputType.TYPE_CLASS_TEXT);
        builder.setView(userInput);
        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                feedbackInput = userInput.getText().toString();
                /* TODO: Submit corrected label with intermediate tensor (feedbackInput) */
                Toast toast = Toast.makeText(getApplicationContext(), String.format("(%s) Thanks for your feedback :)", feedbackInput), Toast.LENGTH_SHORT);
                Log.d(tag, String.format("Correct output: %s", feedbackInput));
                toast.show();
                finish();
            }
        });
        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.cancel();
            }
        });
 
        /* Pull & push */
        button = (Button) findViewById(R.id.pull_model_params);
        button.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(tag, "Update button triggered");
                pullUpdatedParams();
                createClassifier(model, device, numThreads);
            }
        });
        button = (Button) findViewById(R.id.push_features);
        button.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(tag, "Push button triggered");
                pushIntermediateFeature();
            }
        });
    }

    /* Inference */
    private void runInference() {
        Log.d(tag, "Inference button triggered");
        textView.setText("Processing...");
        /* Reshaping image */
        Bitmap scaledImage = Bitmap.createScaledBitmap(image, imageSizeX, imageSizeY, false);

        if (classifier != null) {
            final long startTime = SystemClock.uptimeMillis();
            final String result = classifier.recognizeImage(scaledImage, orientation, MainActivity.context);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            textView.setText(String.format("%s", result));
            feedbackInput = String.format("%s", result);
        }
        correct.setVisibility(View.VISIBLE);
        wrong.setVisibility(View.VISIBLE);
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
            Log.e(tag, Log.getStackTraceString(e));
        }

        imageSizeX = classifier.getImageSizeX();
        imageSizeY = classifier.getImageSizeY();
    }

    private void pullUpdatedParams() {
        try {
            String response = new HttpTask().execute("get").get();
            if (response != null) {
                classifier.close();
                FileOutputStream fos = openFileOutput(classifier.getModelPath(), MODE_PRIVATE);
                byte[] content = response.getBytes("ISO-8859-1");
                fos.write(content);
                fos.flush();
                fos.close();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void pushIntermediateFeature() {
        try {
            String response = new HttpTask().execute("post").get();
            if (response != null) {
                Log.d(tag, "HTTP Response: " + response);
                JSONObject json = new JSONObject(response);
                Log.d(tag, "success: " + json.getString("success"));
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }
}
