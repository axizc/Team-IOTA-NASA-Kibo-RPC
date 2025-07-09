package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.aruco.Aruco;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.aruco.Dictionary;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import static java.lang.Thread.sleep;


/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
    private ObjectDetector objectDetector;
    private void initDetector(Context context) {
        try {
            ObjectDetector.ObjectDetectorOptions options = ObjectDetector.ObjectDetectorOptions.builder()
                    .setMaxResults(5)
                    .setScoreThreshold(0.4f)
                    .build();

            objectDetector = ObjectDetector.createFromFileAndOptions(
                    context,
                    "newcv2.tflite",  // must be in src/main/assets/
                    options
            );
        } catch (Exception e) {
            Log.e("TFLiteInit", "Failed to initialize ObjectDetector", e);
        }
    }
    private Bitmap matToBitmap(Mat mat) {
        Bitmap bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        org.opencv.android.Utils.matToBitmap(mat, bmp);
        return bmp;
    }
    public static Mat undistortImage(Mat image){
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        double[] NAVCAM_CAMERA_MATRIX_SIMULATION = new double[]{523.10575D, 0.0D, 635.434258D, 0.0D, 534.765913D, 500.335102D, 0.0D, 0.0D, 1.0D};
        cameraMatrix.put(0, 0, NAVCAM_CAMERA_MATRIX_SIMULATION);
        Mat cameraCoefficients = new Mat(1,5,CvType.CV_64F);
        double[] NAVCAM_DISTORTION_COEFFICIENTS_SIMULATION = new double[]{-0.164787D, 0.020375D, -0.001572D, -3.69E-4D, 0.0D};

        cameraCoefficients.put(0,0,NAVCAM_DISTORTION_COEFFICIENTS_SIMULATION );
        Mat undistortImg = new Mat();
        Calib3d.undistort(image, undistortImg, cameraMatrix, cameraCoefficients);

        return undistortImg;
    };
    public static List<Mat> detectAruco(Mat undistortImg){
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat markerIds = new Mat();
        Aruco.detectMarkers(undistortImg, dictionary, corners, markerIds);

        return corners;
    };
    public static Mat detectArucoIds(Mat undistortImg){
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat markerIds = new Mat();
        Aruco.detectMarkers(undistortImg, dictionary, corners, markerIds);

        return markerIds;
    };

    public static void printListMat(List<Mat> corners){
        for (int i = 0; i < corners.size(); i++) {
            Mat mat = corners.get(i);
            System.out.println("Marker " + i + " corners:");
            for (int row = 0; row < mat.rows(); row++) {
                System.out.println("row"+row);
                for (int col = 0; col < mat.cols(); col++) {
                    double[] point = mat.get(row, col);
                    if (point != null) {
                        System.out.println("col"+col);
                        System.out.println("Point (" + col + "): x=" + point[0] + ", y=" + point[1]);
                    }
                }
            }
            System.out.println("----");
        }
    }
    public static int caseFind(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4){
        int casse=0;
        double slope=0;
        if (Math.abs(y1-y2)<Math.abs(x1-x2)) {
            if (Math.abs(x1 - x4) < Math.abs(Math.max(x1, x4) - Math.min(x3, x2)) && Math.abs(y1 - y2) < Math.abs(Math.max(y1, y2) - Math.min(y3, y4)) && x1 + x4 < x2 + x3 && y1 + y2 < y3 + y4) {
                casse = 1;

            } else if (x1 + x4 > x2 + x3 && y1 + y2 > y3 + y4 && Math.abs(x1 - x4) < Math.abs(Math.max(x1, x4) - Math.min(x3, x2)) && Math.abs(y1 - y2) < Math.abs(Math.max(y1, y2) - Math.min(y3, y4))) {
                casse = 3;
            }
        }
        if (Math.abs(y1-y2)>Math.abs(x1-x2)) {

            if (x1 + x2 < x3 + x4 && y2 + y3 < y1 + y4 && Math.abs(x1 - x2) < Math.abs(Math.max(x1, x2) - Math.min(x3, x4)) && Math.abs(y1 - y4) < Math.abs(Math.max(y1, y4) - Math.min(y3, y2))) {
                casse = 2;
            } else if (x1 + x2 > x3 + x4 && y2 + y3 > y1 + y4 && Math.abs(x1 - x2) < Math.abs(Math.max(x1, x2) - Math.min(x3, x4)) && Math.abs(y1 - y4) < Math.abs(Math.max(y1, y4) - Math.min(y3, y2))) {
                casse = 4;

            }
        }
        System.out.println(casse);
        return casse;
    }
    public static Mat rotateImage(Mat undistortImg, double rotation_degrees){
        org.opencv.core.Point center = new org.opencv.core.Point((double) undistortImg.width() /2, (double) undistortImg.height() /2);
        Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, rotation_degrees, 1);
        Size size = new Size(undistortImg.width(), undistortImg.height());
        Mat rotated_image= new Mat();
        Imgproc.warpAffine(undistortImg,rotated_image, rotationMatrix, size);
        Imgcodecs.imwrite("rotated_image.png", rotated_image);

        return rotated_image;
    }
    public static Mat cropRegion(Mat src, int x, int y, int w, int h) {
        if (x+w>1280){
            w=1280-x;
        }
        if (x<0){
            x=0;
        }
        if (y<0){
            y=0;
        }
        if (y+h>960){
            y=960-h;
        }
        Rect roi = new Rect(x, y, w, h);
        Mat cropped = new Mat(src, roi);
        Mat independent = cropped.clone();
        Imgcodecs.imwrite("my_crop.png", independent);

        return independent;
    }
    public static double ratio(double casse,double x1,double x2,double x3,double x4){
        double ratio=0;
        if (casse==1){
            ratio=Math.abs(x1-x2)/5;
        }
        else if (casse==2){
            ratio=Math.abs(x3-x1)/5;
        }
        else if (casse==3){
            ratio=Math.abs(x1-x2)/5;
        }
        else if (casse==4){
            ratio=Math.abs(x1-x4)/5;
        }
        else{
            ratio = 8;
        }
        return ratio;
    }
    public static double GreatestandSecondGreatestDiff(double[] arr) {
        if (arr == null || arr.length < 2) {
            return 0;
        }

        double largest = 0;
        double secondLargest = 0;

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] >= largest) {
                secondLargest = largest;
                largest = arr[i];
            } else if (arr[i] > secondLargest && arr[i] != largest) {

                secondLargest = arr[i];
            }
        }

        return largest-secondLargest;
    }
    public List<String> detectObjects(Bitmap bitmap, Context context) {
        List<String> results = new ArrayList<>();

        try {
            AssetFileDescriptor fileDescriptor = context.getAssets().openFd("cv3.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

            Interpreter interpreter = new Interpreter(modelBuffer);

            int inputSize = 256;
            Bitmap resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);
            ByteBuffer input = ByteBuffer.allocateDirect(1 * 3 * inputSize * inputSize * 4);
            Log.e("TFLite", "Allocated inputBuffer size: " + input.capacity());

            input.order(ByteOrder.nativeOrder());

            int[] pixels = new int[inputSize * inputSize];
            resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize);
            for (int y = 0; y <inputSize; y++) {
                for (int x = 0; x < inputSize; x++) {
                    int pixel = pixels[y * inputSize + x];
                    input.putFloat(((pixel >> 16) & 0xFF) / 255.0f);
                    input.putFloat(((pixel >> 8) & 0xFF) / 255.0f);
                    input.putFloat((pixel & 0xFF) / 255.0f);
                }
            }

            float[][][] output = new float[1][300][6];
            interpreter.run(input, output);

            String[] labels = {
                    "coin", "compass", "coral", "crystal", "diamond", "emerald",
                    "fossil", "key", "letter", "shell", "treasure_box"
            };

            for (int i = 0; i < 300; i++) {
                float score = output[0][i][4];
                int classId = (int) output[0][i][5];

                String label = labels[classId];

                Log.e("YOLOv8-256",label+ score * 100);

                if (score < 0.7f) continue;

                classId = (int) output[0][i][5];
                if (classId < 0 || classId >= labels.length) continue;

                 label = labels[classId];
                float x1 = output[0][i][0];
                float y1 = output[0][i][1];
                float x2 = output[0][i][2];
                float y2 = output[0][i][3];


                results.add(label);
            }

        } catch (Exception e) {
            Log.e("TFLiteDetection", "Detection failed", e);
        }

        return results;
    }
    public Mat cropImageFull(Mat image, int index){
            Mat undistortImg = undistortImage(image); //TODO CHANGE IT TO THE ACTUAL PARAMETERS
        try {
            List<Mat> corners = detectAruco(undistortImg);
            printListMat(corners);
            double slope = 0;
            int i = 0;
            int casse = 0;
            double rotation_degrees = 0;
            System.out.println(i);
            Mat detect = corners.get(0);
            double x1 = detect.get(0, 0)[0];
            double y1 = detect.get(0, 0)[1];
            double x2 = detect.get(0, 1)[0];
            double y2 = detect.get(0, 1)[1];

            double xsquare = Math.pow(x2 - x1, 2);
            double ysquare = Math.pow(y2 - y1, 2);
            double distance = Math.sqrt(xsquare + ysquare);
            double cosarg = Math.max(Math.abs(y2 - y1), Math.abs(x2 - x1));
            System.out.println("y1-y2= " + (y1 - y2) + " x1-x2= " + (x1 - x2));

            double rot_degrees1= Math.toDegrees(Math.acos(cosarg / distance));
            double rot_degrees2 = 360- Math.toDegrees(Math.acos(cosarg / distance));


            Mat rotated_image= rotateImage(undistortImg,rot_degrees1);
            corners= detectAruco(rotated_image);
            detect = corners.get(0);
            x1= detect.get(0,0)[0];
            y1= detect.get(0,0)[1];
            x2= detect.get(0,1)[0];
            y2= detect.get(0,1)[1];
            double x3= detect.get(0,2)[0];
            double y3= detect.get(0,2)[1];
            double x4= detect.get(0,3)[0];
            double y4= detect.get(0,3)[1];
            double[] first_dect=new double[]{x1,x2,x3,x4};

            Mat rotated_image2= rotateImage(undistortImg,rot_degrees2);
            List<Mat> corners2= detectAruco(rotated_image2);
            Mat detect2 = corners2.get(0);
            double x12= detect2.get(0,0)[0];
            double y12= detect2.get(0,0)[1];
            double x22= detect2.get(0,1)[0];
            double y22= detect2.get(0,1)[1];
            double x32= detect2.get(0,2)[0];
            double y32= detect2.get(0,2)[1];
            double x42= detect2.get(0,3)[0];
            double y42= detect2.get(0,3)[1];
            double[] sec_dect= new double[]{x12,x22,x32,x42};
            if (GreatestandSecondGreatestDiff(sec_dect)<GreatestandSecondGreatestDiff(first_dect)){
                detect= detect2;
                x1=x12;
                y1=y12;
                x2=x22;
                y2=y22;
                x3=x32;
                y3=y32;
                x4=x42;
                y4=y42;
                rotated_image=rotated_image2;
            }
                casse = caseFind(x1, y1, x2, y2, x3, y3, x4, y4);
                System.out.println(casse);
                double ratio= distance/5;
                Mat cropImage=new Mat();
            if (casse==1){
                double x = x2-30*ratio;
                double y = y2-7*ratio;
                cropImage= cropRegion(rotated_image, (int) x, (int) y, (int) (25*ratio), (int) (27*ratio));
            }
            else if (casse==2){
                double x = x1-3.5*ratio;
                double y = y1;
                cropImage= cropRegion(rotated_image, (int) x, (int) y, (int) (20*ratio), (int) (25*ratio));
            }
            else if (casse==3){
                double x = x1;
                double y = y1-18*ratio;
                cropImage= cropRegion(rotated_image, (int) x, (int) y, (int) (26*ratio), (int) (27*ratio));
            }
            else if (casse==4){
                double x = x4-15*ratio;
                double y = y4-23*ratio;
                cropImage =cropRegion(rotated_image, (int) x, (int) y, (int) (27*ratio), (int) (23.5*ratio));
            }

            else{
                double x = x4-20*ratio;
                double y = y4-10*ratio;
                cropImage=cropRegion(rotated_image, (int) x, (int) y, (int) (27*ratio), (int) (27*ratio));
            }

                api.saveMatImage(cropImage, "my_crop" + index + ".png");
                api.saveMatImage(undistortImg, "output_with_boxes" + index + ".png");
                return cropImage;

        }catch (Exception e){
            Log.e("TFLite", "Allocated inputBuffer size: " , e);
            return undistortImg;
        }
    }
    public double distanceAwayX(Mat image){
        Mat undistortImg = undistortImage(image); //TODO CHANGE IT TO THE ACTUAL PARAMETERS
        try {
            List<Mat> corners = detectAruco(undistortImg);
            printListMat(corners);

            Mat detect = corners.get(0);
            double x1= detect.get(0,0)[0];
            double y1= detect.get(0,0)[1];
            double x2= detect.get(0,1)[0];
            double y2= detect.get(0,1)[1];
            double x3= detect.get(0,2)[0];
            double y3= detect.get(0,2)[1];
            double x4= detect.get(0,3)[0];
            double y4= detect.get(0,3)[1];
            double avgx= (x1+x2+x3+x4)/4;
            double distance = Math.sqrt(Math.pow(x2-x1,2)+Math.pow(y2-y1,2));
            double meterConvert = distance/0.05;
            return (1260/2-avgx)/meterConvert;
        }

        catch(Exception e){
            Log.e("Error", "failed", e);
            return 0;
        }

    }
    public double distanceAwayY(Mat image){
        Mat undistortImg = undistortImage(image); //TODO CHANGE IT TO THE ACTUAL PARAMETERS
        try {
            List<Mat> corners = detectAruco(undistortImg);
            printListMat(corners);

            Mat detect = corners.get(0);
            double x1= detect.get(0,0)[0];
            double y1= detect.get(0,0)[1];
            double x2= detect.get(0,1)[0];
            double y2= detect.get(0,1)[1];
            double x3= detect.get(0,2)[0];
            double y3= detect.get(0,2)[1];
            double x4= detect.get(0,3)[0];
            double y4= detect.get(0,3)[1];
            double avgy= (y1+y2+y3+y4)/4;
            double distance = Math.sqrt(Math.pow(x2-x1,2)+Math.pow(y2-y1,2));
            double meterConvert = distance/0.05;
            return (960/2-avgy)/meterConvert;
        }

        catch(Exception e){
            Log.e("Error", "failed", e);
            return 0;
        }

    }
    public String landmarkGetter(List<String> detections){

        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */
        List<String> landmarks = new ArrayList<String>();
        List<String> treasure_item = new ArrayList<String>();
        landmarks.add("coin");
        landmarks.add("compass");
        landmarks.add("coral");
        landmarks.add("fossil");
        landmarks.add("key");
        landmarks.add("letter");
        landmarks.add("shell");
        landmarks.add("treasure_box");
        treasure_item.add("crystal");
        treasure_item.add("diamond");
        treasure_item.add("emerald");
        String area1treasure ="";
        int quantityOfLand= 0;
        String landType= "";
        for (int i=0; i<detections.size(); ++i){
            Log.e("Detection",detections.get(i));
            if(landmarks.contains(detections.get(i))){
                if (landType =="" || detections.get(i)==landType) {
                    quantityOfLand++;
                    landType = detections.get(i);
                    Log.e("Tflite landmark", quantityOfLand + landType);
                }
            }
            else if(treasure_item.contains(detections.get(i))){
                if (area1treasure== "") {
                    area1treasure = detections.get(i);
                    Log.e("Tflite treasure", quantityOfLand + landType);
                }
            }
        }
        return landType;
    }
    public String treasureGetter(List<String> detections){

        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */

        List<String> landmarks = new ArrayList<String>();
        List<String> treasure_item = new ArrayList<String>();
        landmarks.add("coin");
        landmarks.add("compass");
        landmarks.add("coral");
        landmarks.add("fossil");
        landmarks.add("key");
        landmarks.add("letter");
        landmarks.add("shell");
        landmarks.add("treasure_box");
        treasure_item.add("crystal");
        treasure_item.add("diamond");
        treasure_item.add("emerald");
        String treasure ="";
        int quantityOfLand= 0;
        String landType= "";
        for (int i=0; i<detections.size(); ++i){
            Log.e("Detection",detections.get(i));
            if(landmarks.contains(detections.get(i))){
                if (landType =="" || detections.get(i)==landType) {
                    quantityOfLand++;
                    landType = detections.get(i);
                    Log.e("Tflite landmark", quantityOfLand + landType);
                }
            }
            else if(treasure_item.contains(detections.get(i))){
                if (treasure== "") {
                    treasure = detections.get(i);
                    Log.e("Tflite treasure", quantityOfLand + landType);
                }
            }
        }
        return treasure;
    }
    public int landmarkQuantity(List<String> detections){

        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */
        List<String> landmarks = new ArrayList<String>();
        List<String> treasure_item = new ArrayList<String>();
        landmarks.add("coin");
        landmarks.add("compass");
        landmarks.add("coral");
        landmarks.add("fossil");
        landmarks.add("key");
        landmarks.add("letter");
        landmarks.add("shell");
        landmarks.add("treasure_box");
        treasure_item.add("crystal");
        treasure_item.add("diamond");
        treasure_item.add("emerald");
        String treasure ="";
        int quantityOfLand= 0;
        String landType= "";
        for (int i=0; i<detections.size(); ++i){
            Log.e("Detection",detections.get(i));
            if(landmarks.contains(detections.get(i))){
                if (landType =="" || detections.get(i)==landType) {
                    quantityOfLand++;
                    landType = detections.get(i);
                    Log.e("Tflite landmark", quantityOfLand + landType);
                }
            }
            else if(treasure_item.contains(detections.get(i))){
                if (treasure== "") {
                    treasure = detections.get(i);
                    Log.e("Tflite treasure", quantityOfLand + landType);
                }
            }
        }
        return quantityOfLand;
    }
    @Override
    protected void runPlan1(){
        initDetector(getApplicationContext());
        // The mission starts.
        api.startMission();

        // Move to a point.
        Point point = new Point(10.9d, -9.92284d, 5.195d);
        Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
        api.moveTo(point, quaternion, false);

        // Get a camera image.
        Mat image1 = api.getMatNavCam();
        Mat image = api.getMatNavCam();
        Mat cropImage = cropImageFull(image1, 0);
        Bitmap inputBitmap = matToBitmap(cropImage);
        List<String> detections= detectObjects(inputBitmap, getApplicationContext());
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        api.saveMatImage(image, "imagee.png");
        cropImage = cropImageFull(image, 4);
        inputBitmap = matToBitmap(cropImage);
        detections= detectObjects(inputBitmap, getApplicationContext());

        detections= detectObjects(inputBitmap, getApplicationContext());
        image = api.getMatNavCam();
        cropImage = cropImageFull(image, 1);
        inputBitmap = matToBitmap(cropImage);
        detections= detectObjects(inputBitmap, getApplicationContext());

        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */

        String area1Treasure= treasureGetter(detections);
        String area1LandType = landmarkGetter(detections);
        int quantityOfLandArea1= landmarkQuantity(detections);
        // When you recognize landmark items, let’s set the type and number.
        api.setAreaInfo(1, area1LandType , quantityOfLandArea1);

        /* **************************************************** */
        /* Let's move to each area and recognize the items. */
        /* **************************************************** */
        point= new Point(10.9d, -8.9d, 4.4d);
        quaternion = new Quaternion( -0.707f, 0f, 0.707f, 0f);
        api.moveTo(point,quaternion,false);
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        api.saveMatImage(image, "imagee.png");
        cropImage = cropImageFull(image, 1);
        inputBitmap = matToBitmap(cropImage);
        detections= detectObjects(inputBitmap, getApplicationContext());

        detections= detectObjects(inputBitmap, getApplicationContext());
        image = api.getMatNavCam();

        Mat image2 = api.getMatNavCam();
        api.saveMatImage(image2,"area2.png");
        cropImage = cropImageFull(image, 1);
        inputBitmap = matToBitmap(cropImage);
        detections= detectObjects(inputBitmap, getApplicationContext());
        String area2Treasure= treasureGetter(detections);
        String area2LandType = landmarkGetter(detections);
        int quantityOfLandArea2= landmarkQuantity(detections);
        // When you recognize landmark items, let’s set the type and number.
        api.setAreaInfo(2, area2LandType , quantityOfLandArea2);

        point= new Point(10.925d, -7.925d, 4.5d);
        quaternion = new Quaternion( -0.707f, 0f, 0.707f, 0f);
        api.moveTo(point,quaternion,false);

        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        api.saveMatImage(image, "image4e.png");
        cropImage = cropImageFull(image, 2);
        inputBitmap = matToBitmap(cropImage);
        detections= detectObjects(inputBitmap, getApplicationContext());

        detections= detectObjects(inputBitmap, getApplicationContext());
        Mat image3 = api.getMatNavCam();

        cropImage = cropImageFull(image3, 2);
        inputBitmap = matToBitmap(cropImage);
        detections= detectObjects(inputBitmap, getApplicationContext());
        String area3Treasure= treasureGetter(detections);
        String area3LandType = landmarkGetter(detections);
        int quantityOfLandArea3= landmarkQuantity(detections);
        // When you recognize landmark items, let’s set the type and number.
        api.setAreaInfo(3, area3LandType , quantityOfLandArea3);
        api.setAreaInfo(3, area3LandType , quantityOfLandArea3);

        point= new Point(10.50d, -6.8525d, 4.965d);
        quaternion = new Quaternion( 0.0f, 0f, -0.991f, 0.131f);
        api.moveTo(point,quaternion,false);

        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        api.saveMatImage(image, "imagee.png");
        cropImage = cropImageFull(image, 3);
        inputBitmap = matToBitmap(cropImage);
        detections= detectObjects(inputBitmap, getApplicationContext());

        detections= detectObjects(inputBitmap, getApplicationContext());
        Mat image4 = api.getMatNavCam();

        cropImage = cropImageFull(image4, 3);
        inputBitmap = matToBitmap(cropImage);
        detections= detectObjects(inputBitmap, getApplicationContext());
        String area4Treasure= treasureGetter(detections);
        String area4LandType = landmarkGetter(detections);
        int quantityOfLandArea4= landmarkQuantity(detections);
        // When you recognize landmark items, let’s set the type and number.
        api.setAreaInfo(4, area4LandType , quantityOfLandArea4);

        // When you move to the front of the astronaut, report the rounding completion.
        point = new Point(11.143d, -6.7607d, 4.9654d);

        quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.reportRoundingCompletion();

        api.moveTo(point, quaternion, false);

        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        image = api.getMatNavCam();
        api.saveMatImage(image, "img3.png");
        cropImage = cropImageFull(image, 4);
        inputBitmap = matToBitmap(cropImage);
        detections= detectObjects(inputBitmap, getApplicationContext());
        detections= detectObjects(inputBitmap, getApplicationContext());
        detections= detectObjects(inputBitmap, getApplicationContext());
        detections= detectObjects(inputBitmap, getApplicationContext());
        detections= detectObjects(inputBitmap, getApplicationContext());
        detections= detectObjects(inputBitmap, getApplicationContext());
        detections= detectObjects(inputBitmap, getApplicationContext());
        detections= detectObjects(inputBitmap, getApplicationContext());
        detections= detectObjects(inputBitmap, getApplicationContext());
        image = api.getMatNavCam();
        api.saveMatImage(image, "img3.png");
        cropImage = cropImageFull(image, 4);
        inputBitmap = matToBitmap(cropImage);
        detections= detectObjects(inputBitmap, getApplicationContext());
        List<String> landmarks = new ArrayList<String>();
        List<String> treasure_item = new ArrayList<String>();
        landmarks.add("coin");
        landmarks.add("compass");
        landmarks.add("coral");
        landmarks.add("fossil");
        landmarks.add("key");
        landmarks.add("letter");
        landmarks.add("shell");
        landmarks.add("treasure_box");
        treasure_item.add("crystal");
        treasure_item.add("diamond");
        treasure_item.add("emerald");
        String treasure ="";
        int quantityOfLand= 0;
        String landType1= "";
        String landType2= "";
        for (int i=0; i<detections.size(); ++i){
            Log.e("Detection",detections.get(i));
            if(landmarks.contains(detections.get(i))){
                quantityOfLand++;
                if (landType1 == landType2) {
                    landType1 = detections.get(i);
                    Log.e("Tflite landmark", quantityOfLand + landType1);
                }
                else {
                    landType2= detections.get(i);
                    Log.e("Tflite landmark", quantityOfLand + landType2);

                }

            }
            else if(treasure_item.contains(detections.get(i))){
                treasure= detections.get(i);
                Log.e("Tflite treasure",quantityOfLand+landType1);

            }
        }


        if (treasure !=""){
            if (treasure==area1Treasure){
                double xinc=0;
                double zinc = 0;
                if (distanceAwayX(image2)<0) {
                    xinc=-0.10;
                }
                else if (distanceAwayX(image1)>0){
                    xinc=0.10;
                }
                if (distanceAwayY(image1)<0){
                    zinc+=0.10;
                }
                else{
                    zinc-=0.10;
                }
                point = new Point((10.9+xinc), -9.92284d, (5.195+zinc));

                quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
                api.moveTo(point, quaternion, false);

            }
            else if (treasure== area2Treasure){
                if (distanceAwayX(image2)<0){
                    point= new Point(10.925d, -8.8d, 4.32d);}
                else{
                    point= new Point(10.925d, -9d, 4.32d);
                }
                quaternion = new Quaternion( -0.707f, 0f, 0.707f, 0f);
                api.moveTo(point,quaternion,false);

            }
            else if (treasure == area3Treasure){
                if (distanceAwayX(image3)<0){
                point= new Point(10.925d, -7.8d, 4.32d);}
                else{
                    point= new Point(10.925d, -8.05d, 4.32d);
                }
                quaternion = new Quaternion( -0.707f, 0f, 0.707f, 0f);
                api.moveTo(point,quaternion,false);


            }
            else if (treasure == area4Treasure){
                if (distanceAwayY(image4)<0) {
                    point = new Point(10.44d, -6.8525d, 4.925d);
                }
                else{
                    point = new Point(10.44d, -6.8525d, 4.965d);
                }
                quaternion = new Quaternion( 0.0f, 0f, 1f, 0f);
                api.moveTo(point,quaternion,false);

            }
        }
        api.notifyRecognitionItem();
        api.takeTargetItemSnapshot();

        /* ********************************************************** */
        /* Write your code to recognize which target item the astronaut has. */
        /* ********************************************************** */

        // Let's notify the astronaut when you recognize it.

        /* ******************************************************************************************************* */
        /* Write your code to move Astrobee to the location of the target item (what the astronaut is looking for) */
        /* ******************************************************************************************************* */

        // Take a snapshot of the target item.
    }

    @Override
    protected void runPlan2(){
       // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }

    // You can add your method.
    private String yourMethod(){
        return "your method";
    }
}
