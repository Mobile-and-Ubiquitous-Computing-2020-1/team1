package com.example.client.tflite;

import android.net.http.AndroidHttpClient;
import android.os.AsyncTask;

import com.example.client.tflite.env.Logger;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.FileEntity;
import org.apache.http.util.EntityUtils;

import java.io.File;
import java.io.IOException;

public class HttpTask extends AsyncTask<String, Void, String> {

    private static final Logger LOGGER = new Logger();
    private Exception exception;
    private String BASE_URL = "http://147.46.219.198:40917/";

    protected String doInBackground(String... params) {
        String method = params[0];
        if (method.equals("get")) {
            return pull();
        } else if (method.equals("post")) {
            return push();
        } else {
            return null;
        }
    }

    private String pull() {
        try {
            AndroidHttpClient http = AndroidHttpClient.newInstance("MyApp");
            HttpGet method = new HttpGet(BASE_URL + "pull/");
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

    private String push() {
        try {
            AndroidHttpClient http = AndroidHttpClient.newInstance("MyApp");
            HttpPost method = new HttpPost(BASE_URL + "push/");
            String FILES_DIR = "/data/user/0/com.example.client/files";
            File file = new File(FILES_DIR + "/intermediates");
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