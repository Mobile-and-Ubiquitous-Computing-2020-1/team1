package com.example.client;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.ImageView;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;


public class MainActivity extends AppCompatActivity {
    /* Tag id for logging */
    protected final String tag = MainActivity.class.getSimpleName();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        /* Load image */
        ImageView demo_image = (ImageView)findViewById(R.id.demo_image);
        Bitmap image=((BitmapDrawable)demo_image.getDrawable()).getBitmap();

        /* Load text for displaying result */
        TextView log_text = (TextView)findViewById(R.id.view_result);

        /* Button for triggering inference*/
        Button button = (Button)findViewById(R.id.run_inference);
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
    }
}
