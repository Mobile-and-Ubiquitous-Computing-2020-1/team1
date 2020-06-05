package com.example.client;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;


public class ListActivity extends AppCompatActivity {
    /* Tag id for logging */
    protected final String tag = ListActivity.class.getSimpleName();
    private static Context context;

    private RecyclerView recyclerView;

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
                Log.d(tag, String.format("You have clicked %d", position));
                startActivity(intent);
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
}
