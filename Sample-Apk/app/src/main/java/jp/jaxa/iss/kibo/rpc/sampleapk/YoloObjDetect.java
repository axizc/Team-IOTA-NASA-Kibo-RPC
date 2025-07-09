package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import java.io.IOException;
import java.util.List;

public class YoloObjDetect {
        private ObjectDetector objectDetector;

        public YoloObjDetect(Context context) throws IOException {
            ObjectDetector.ObjectDetectorOptions options =
                    ObjectDetector.ObjectDetectorOptions.builder()
                            .setMaxResults(5)
                            .setScoreThreshold(0.5f)
                            .build();

            objectDetector = ObjectDetector.createFromFileAndOptions(
                    context,
                    "newcv2.tflite",
                    options
            );
        }

        public List<Detection> detect(Bitmap bitmap) {
            TensorImage image = TensorImage.fromBitmap(bitmap);
            return objectDetector.detect(image);
        }

        public void printResults(List<Detection> results) {
            for (Detection detection : results) {
                RectF boundingBox = detection.getBoundingBox();
                String category = detection.getCategories().get(0).getLabel();
                float score = detection.getCategories().get(0).getScore();

                System.out.printf("Detected %s with %.2f confidence at %s\n", category, score, boundingBox);
            }
        }
    }
