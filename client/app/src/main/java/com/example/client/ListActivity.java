package com.example.client;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.client.tflite.Classifier;
import com.example.client.tflite.Classifier.Device;
import com.example.client.tflite.Classifier.Model;
import com.example.client.tflite.HttpTask;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;


public class ListActivity extends AppCompatActivity {
    /* Tag id for logging */
    protected final String tag = ListActivity.class.getSimpleName();
    private static Context context;

    private Model model = Model.FLOAT_MOBILENET;
    private Device device = Device.CPU;
    private int numThreads = -1;
    private int imageSizeX;
    private int imageSizeY;

    public static Classifier classifier;

    private RecyclerView recyclerView;
    private Button pull_button, push_button;

    public ArrayList<ImageItem> createLists;

    /* Variables for image grid */
    private String image_titles[] = {
            "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10",
    };

    private Integer image_ids[] = {
            R.drawable.demo01, R.drawable.demo02, R.drawable.demo03,
            R.drawable.demo04, R.drawable.demo05, R.drawable.demo06,
            R.drawable.demo07, R.drawable.demo08, R.drawable.demo09,
            R.drawable.demo10,
    };

    public class ImageItem {
        private String image_title;
        private Integer image_id;

        public String getImageTitle() { return image_title; }
        public Integer getImageId() { return image_id; }
        public void setImageTitle(String newname) { this.image_title = newname; }
        public void setImageId(Integer newid) { this.image_id = newid; }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ListActivity.context = getApplicationContext();
        setContentView(R.layout.activity_list);
        Log.d(tag, "Connected");

        createClassifier(model, device, numThreads);

        recyclerView = (RecyclerView)findViewById(R.id.imagegallery);
        recyclerView.setHasFixedSize(true);

        RecyclerView.LayoutManager layoutManager = new GridLayoutManager(getApplicationContext(), 2);
        recyclerView.setLayoutManager(layoutManager);
        createLists = prepareData();
        ImageAdapter adapter = new ImageAdapter(getApplicationContext(), createLists);
        recyclerView.setAdapter(adapter);

        adapter.setOnItemClickListener(new ImageAdapter.ClickListener() {
            @Override
            public void onItemClick(int position, View v) {
                Intent intent = new Intent(ListActivity.this, MainActivity.class);
                intent.putExtra("position", image_ids[position]);
                intent.putExtra("imageSizeX", imageSizeX);
                intent.putExtra("imageSizeY", imageSizeY);
                Log.d(tag, String.format("You have clicked %d", position));
                startActivity(intent);
            }
        });

        /* Pull & push */
        pull_button = (Button) findViewById(R.id.pull_model_params);
        pull_button.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(tag, "Update button triggered");
                pullUpdatedParams();
            }
        });
        push_button = (Button) findViewById(R.id.push_features);
        push_button.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(tag, "Push button triggered");
                pushIntermediateFeature();
            }
        });
    }

    private ArrayList<ImageItem> prepareData() {
        ArrayList<ImageItem> imlist = new ArrayList<>();
        for(int i = 0; i < image_titles.length; i++) {
            ImageItem imageitem = new ImageItem();
            imageitem.setImageTitle(image_titles[i]);
            imageitem.setImageId(image_ids[i]);
            imlist.add(imageitem);
        }
        return imlist;
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
                createClassifier(model, device, numThreads);
                Toast toast;
                toast = Toast.makeText(getApplicationContext(), "New model is Updated!", Toast.LENGTH_SHORT);
                toast.show();
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
            File files_dir = getFilesDir();
            for (String filename: files_dir.list()) {
                if (filename.startsWith("intermediates") || filename.startsWith("label")) {
                    File file = new File(files_dir, filename);
                    String response = new HttpTask().execute("post", file.getAbsolutePath()).get();
                    if (response != null) {
                        Log.d(tag, "HTTP Response: " + response);
                        JSONObject json = new JSONObject(response);
                        Log.d(tag, "success: " + json.getString("success"));
                    }
                }
            }
            Toast toast;
            toast = Toast.makeText(getApplicationContext(), "Successfully sent your feedback!", Toast.LENGTH_SHORT);
            toast.show();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }
}
