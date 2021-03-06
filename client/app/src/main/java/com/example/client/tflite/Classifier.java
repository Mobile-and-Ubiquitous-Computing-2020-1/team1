/*
The code is largely borrowed from the official image classification example
of tensorflow-lite repository under the Apache License (v2.0).
 */

package com.example.client.tflite;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;

import com.example.client.tflite.env.Logger;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import static android.content.Context.MODE_APPEND;
import static android.content.Context.MODE_PRIVATE;

/** A classifier specialized to label images using TensorFlow Lite. */
public abstract class Classifier {
    private static final Logger LOGGER = new Logger();

    /** The model type used for classification. */
    public enum Model {
        RESNET,
        FLOAT_MOBILENET,
        FACENET,
    }

    /** The runtime device type used for executing classification. */
    public enum Device {
        CPU,
        NNAPI,
        GPU
    }

    /** Number of results to show in the UI. */
    private static final int MAX_RESULTS = 3;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** Image size along the x axis. */
    private final int imageSizeX;

    /** Image size along the y axis. */
    private final int imageSizeY;

    /** Optional GPU delegate for accleration. */
    private GpuDelegate gpuDelegate = null;

    /** Optional NNAPI delegate for accleration. */
    private NnApiDelegate nnApiDelegate = null;

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /** Labels corresponding to the output of the vision model. */
    public List<String> labels;

    /** Input image TensorBuffer. */
    private TensorImage inputImageBuffer;

    /** Output TensorBuffers */
    private final TensorBuffer[] outputBuffers;

    /** Processer to apply post processing of the output probability. */
    private final TensorProcessor probabilityProcessor;

    /**
     * Creates a classifier with the provided configuration.
     *
     * @param activity The current Activity.
     * @param model The model to use for classification.
     * @param device The device to use for classification.
     * @param numThreads The number of threads to use for classification.
     * @return A classifier with the desired configuration.
     */
    public static Classifier create(Activity activity, Model model, Device device, int numThreads)
            throws IOException {
        if (model == Model.RESNET) {
            return new ClassifierResNet(activity, device, numThreads);
        } else if (model == Model.FLOAT_MOBILENET) {
            return new ClassifierFloatMobileNet(activity, device, numThreads);
        } else if (model == Model.FACENET) {
            return new ClassifierFaceNet(activity, device, numThreads);
        } else {
            throw new UnsupportedOperationException();
        }
    }

    /** An immutable result returned by a Classifier describing what was recognized. */
    public static class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /** Display name for the recognition. */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        public Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            /*
            if (id != null) {
                resultString += "[" + id + "] ";
            }
            */
            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    /** Initializes a {@code Classifier}. */
    protected Classifier(Activity activity, Device device, int numThreads) throws IOException {
        tfliteModel = getTfliteModel(activity);
        switch (device) {
            case NNAPI:
                nnApiDelegate = new NnApiDelegate();
                tfliteOptions.addDelegate(nnApiDelegate);
                break;
            case GPU:
                gpuDelegate = new GpuDelegate();
                tfliteOptions.addDelegate(gpuDelegate);
                break;
            case CPU:
                break;
        }
        tfliteOptions.setNumThreads(numThreads);
        tflite = new Interpreter(tfliteModel, tfliteOptions);
        LOGGER.d("Interpreter initialized");

        // Loads labels out from the label file.
        labels = FileUtil.loadLabels(activity, getLabelPath());
        LOGGER.d("num labels " + labels.size());
        LOGGER.d("first label " + labels.get(0));
        LOGGER.d("last label " + labels.get(labels.size() - 1));

        // Reads type and shape of input and output tensors, respectively.
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];

        LOGGER.d("image SizeY " + imageSizeY);
        LOGGER.d("image SizeX " + imageSizeX);

        outputBuffers = new TensorBuffer[tflite.getOutputTensorCount()];
        for (int outputIdx = 0; outputIdx < tflite.getOutputTensorCount(); outputIdx++) {
            int[] outputShape = tflite.getOutputTensor(outputIdx).shape();
            LOGGER.d(outputIdx + "-th output Shape " + Arrays.toString(outputShape));
            DataType outputDataType = tflite.getOutputTensor(outputIdx).dataType();
            outputBuffers[outputIdx] =  TensorBuffer.createFixedSize(outputShape, outputDataType);
        }

        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);

        // Creates the post processor for the output probability.
        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

        LOGGER.d("Created a Tensorflow Lite Image Classifier.");
    }

    private int argmax(float[] prob) {
        float max = prob[0];
        int idx = 0;
        for (int i = 1; i < prob.length; i++) {
            if (prob[i] > max) {
                max = prob[i];
                idx = i;
            }
        }
        return idx;
    }

    /** Runs inference and returns the classification results. */
    public String recognizeImage(final Bitmap bitmap, int imageId, int sensorOrientation, Context context) {

        // Logs this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("loadImage");
        long startTimeForLoadImage = SystemClock.uptimeMillis();
        inputImageBuffer = loadImage(bitmap, sensorOrientation);
        long endTimeForLoadImage = SystemClock.uptimeMillis();
        Trace.endSection();
        LOGGER.v("Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage));

        // Runs the inference call.
        Trace.beginSection("runInference");
        Object[] inputs = {inputImageBuffer.getBuffer()};
        Map<Integer, Object> outputs = new HashMap<Integer, Object>();
        for (int i = 0; i < outputBuffers.length; i++) {
            outputs.put(i, outputBuffers[i].getBuffer().rewind());
        }
        long startTimeForReference = SystemClock.uptimeMillis();
        tflite.runForMultipleInputsOutputs(inputs, outputs);
        long endTimeForReference = SystemClock.uptimeMillis();
        Trace.endSection();
        LOGGER.v("Timecost to run model inference: " + (endTimeForReference - startTimeForReference));

        FileOutputStream fos = null;
        try {
            String filename = "intermediates_" + imageId;
            File file = new File(context.getFilesDir(), filename);
            if (!file.exists()) {
                fos = context.openFileOutput(filename, MODE_PRIVATE);
                FileChannel fc = fos.getChannel();
                outputBuffers[1].getBuffer().rewind();
                fc.write(outputBuffers[1].getBuffer());
                fc.close();
            }
        } catch (FileNotFoundException e) {
            LOGGER.v("failed to write ByteBuffer (FileNotFoundException) " + e);
            e.printStackTrace();
        } catch (IOException e ) {
            LOGGER.v("failed to write ByteBuffer (IOException) " + e);
            e.printStackTrace();
        } catch (IndexOutOfBoundsException e) {
            LOGGER.v("failed to write ByteBuffer (IndexOutOfBoundsException) " + e);
            e.printStackTrace();
        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    // this will not be catches (already check whether fos is null)
                    e.printStackTrace();
                }
            }
        }
        LOGGER.v("writing intermediates finished");

        float[] probs = outputBuffers[0].getFloatArray();
        int idx = argmax(probs);

        LOGGER.d("argmax idx " + idx + " " + labels.get(idx));
        return labels.get(idx);
    }

    /** Closes the interpreter and model to release resources. */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
        tfliteModel = null;
    }

    /** Get the image size along the x axis. */
    public int getImageSizeX() {
        return imageSizeX;
    }

    /** Get the image size along the y axis. */
    public int getImageSizeY() {
        return imageSizeY;
    }

    /** Loads input image, and applies preprocessing. */
    private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRotation = sensorOrientation / 90;
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new Rot90Op(numRotation))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    /** Gets the top-k results. */
    private static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue(), null));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    /** Gets the name of the model file stored in Assets. */
    public abstract String getModelPath();

    /** Gets the name of the label file stored in Assets. */
    protected abstract String getLabelPath();

    /** Gets the TensorOperator to nomalize the input image in preprocessing. */
    protected abstract TensorOperator getPreprocessNormalizeOp();

    /**
     * Gets the TensorOperator to dequantize the output probability in post processing.
     *
     * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
     * essentially linear transformation). For float model, de-quantize is not required. But to
     * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
     * 1.0f, respectively.
     */
    protected abstract TensorOperator getPostprocessNormalizeOp();

    private MappedByteBuffer getTfliteModel(Activity activity) throws IOException {
        try {
            File modelFile = new File(activity.getFilesDir(), getModelPath());
            if (!modelFile.exists()) {
                FileOutputStream fos = activity.openFileOutput(getModelPath(), MODE_PRIVATE);
                InputStream is = activity.getAssets().open(getModelPath());
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, bytesRead);
                }
                fos.flush();
                fos.close();
                is.close();
            }
            FileInputStream is = activity.openFileInput(getModelPath());
            return is.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, modelFile.length());
        } catch (IOException e) {
            e.printStackTrace();
            throw e;
        }
    }
}