package com.example.client.tflite;

import android.net.http.AndroidHttpClient;
import android.os.AsyncTask;

import com.example.client.tflite.env.Logger;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.FileEntity;
import org.apache.http.util.EntityUtils;

import java.io.File;
import java.io.IOException;

public class PushFeatureTask extends AsyncTask<String, Void, String> {

    private static final Logger LOGGER = new Logger();
    private Exception exception;

    protected String doInBackground(String... urls) {
        try {
            AndroidHttpClient http = AndroidHttpClient.newInstance("MyApp");
            HttpPost method = new HttpPost(urls[0]);
            File file = new File("/data/user/0/com.example.client/files/intermediates");
            method.setEntity(new FileEntity(file, "application/octet-stream"));
            HttpResponse response = http.execute(method);
            HttpEntity entity = response.getEntity();
            String s = EntityUtils.toString(entity);
            http.close();
            return s;
        } catch (IOException e) {
            LOGGER.d("Http failed");
            e.printStackTrace();
            this.exception = e;
            return null;
        }
    }
}