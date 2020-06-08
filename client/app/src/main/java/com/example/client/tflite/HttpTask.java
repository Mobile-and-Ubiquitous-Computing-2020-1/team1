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
            String filePathpath = params[1];
            return push(filePathpath);
        } else {
            return null;
        }
    }

    private String pull() {
        try {
            AndroidHttpClient http = AndroidHttpClient.newInstance("MyApp");
            HttpGet method = new HttpGet(BASE_URL + "model/best/pull");
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

    private String info() {
        try {
            AndroidHttpClient http = AndroidHttpClient.newInstance("MyApp");
            HttpGet method = new HttpGet(BASE_URL + "model/best/info");
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

    private String push(String filePath) {
        try {
            AndroidHttpClient http = AndroidHttpClient.newInstance("MyApp");
            HttpPost method = new HttpPost(BASE_URL + "feature/push");
            File file = new File(filePath);
            method.setEntity(new FileEntity(file, "application/octet-stream;"));
            method.setHeader("Content-Disposition", String.format("form-data; filename=\"%s\"", file.getName()));
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