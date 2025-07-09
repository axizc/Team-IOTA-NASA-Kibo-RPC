import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;


import org.opencv.aruco.Aruco;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;

import org.opencv.aruco.Dictionary;

import org.opencv.objdetect.Objdetect;

import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

public class Main{

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
            System.out.println("yep");
            if (Math.abs(x1 - x4) < Math.abs(Math.max(x1, x4) - Math.min(x3, x2)) && Math.abs(y1 - y2) < Math.abs(Math.max(y1, y2) - Math.min(y3, y4)) && x1 + x4 < x2 + x3 && y1 + y2 < y3 + y4) {
                casse = 1;
                System.out.println("1");

            } else if (x1 + x4 > x2 + x3 && y1 + y2 > y3 + y4 && Math.abs(x1 - x4) < Math.abs(Math.max(x1, x4) - Math.min(x3, x2)) && Math.abs(y1 - y2) < Math.abs(Math.max(y1, y2) - Math.min(y3, y4))) {
                casse = 3;
                System.out.println("3");
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
        Point center = new Point((double) undistortImg.width() /2, (double) undistortImg.height() /2);
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
            System.out.println("Array must contain at least two elements.");
            return 0;
        }

        double largest = 0;
        double secondLargest = 0;

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] >= largest) {
                secondLargest = largest; // The old largest becomes the new second largest
                largest = arr[i];        // The current element is the new largest
            } else if (arr[i] > secondLargest && arr[i] != largest) {
                // If the current element is greater than secondLargest but not equal to largest
                secondLargest = arr[i];
            }
        }

            return largest-secondLargest;
    }
    public static Mat YoloOnnXDetector (Mat image) {
        final String[] CLASS_NAMES = new String[]{
                "coin", "compass", "coral", "crystal", "diamond", "emerald", "fossil", "key", "letter", "shell", "treasure_box"
        };
        String modelPath = "C:\\Users\\vt\\runs\\detect\\yolov8n_custom6\\weights\\best.onnx";
        String imgPath = "1.png";
        Net net = Dnn.readNetFromONNX(modelPath);
//        Mat mat= Imgcodecs.imread(imgPath);
        Mat mat = image;
        Mat blob = Dnn.blobFromImage(mat,1/255.0, new Size(128,128));
        net.setInput(blob);
        Mat predict = net.forward();
        Mat mask= predict.reshape(0,1).reshape(0, predict.size(1));
        double width = mat.cols()/128.0;
        double height = mat.rows()/128.0;
        Rect2d[] rect2d = new Rect2d[mask.cols()];
        float[] scoref = new float[mask.cols()];
        int[] classid = new int[mask.cols()];
        for(int i=0;i<mask.cols();i++){
            double[] x = mask.col(i).get(0,0);
            double[] y = mask.col(i).get(1,0);
            double[] w = mask.col(i).get(2,0);
            double[]  h = mask.col(i).get(3,0);
            rect2d[i] = new Rect2d((x[0]-w[0]/2)*width, (y[0]-h[0]/2)*height, w[0]*width, h[0]*height);
            Mat score = mask.col(i).submat(4, predict.size(1)-1, 0, 1);
            Core.MinMaxLocResult mmr = Core.minMaxLoc(score);
            scoref[i] = (float)mmr.maxVal;
            classid[i]= (int) mmr.maxLoc.y;
        }
        MatOfRect2d bboxes = new MatOfRect2d(rect2d);
        MatOfFloat scores = new  MatOfFloat(scoref);
        MatOfInt indices = new  MatOfInt();
        Dnn.NMSBoxes(bboxes, scores, 0.5f, 0.5f, indices);
        List<Integer> result =  indices.toList();
        for (Integer integer : result) {
            Imgproc.rectangle(mat, new Rect(rect2d[integer].tl(), rect2d[integer].size()),
                    new Scalar(255,0,0),1);
                    Imgproc.putText(mat, classid[integer]+":"+scoref[integer], rect2d[integer].tl(),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0,255,0));

        }
        Imgcodecs.imwrite("result.png", mat);
        return mat;

    }
    public static void main(String[] args) {
         System.load("C:\\Users\\vt\\Downloads\\sai3\\bld\\install\\java\\opencv_java453.dll");

        Mat image = Imgcodecs.imread("C:\\Users\\vt\\Downloads\\output_with_boxes1.png");
        Mat undistortImg = Imgcodecs.imread("C:\\Users\\vt\\Downloads\\output_with_boxes2.png");
        List<Mat> corners = detectAruco(undistortImg);
        printListMat(corners);
        double slope = 0;
        int i = 0;
        int casse = 0;
        double rotation_degrees = 0;
            System.out.println(i);
        Mat detect = corners.get(0);
        double x1= detect.get(0,0)[0];
        double y1= detect.get(0,0)[1];
        double x2= detect.get(0,1)[0];
        double y2= detect.get(0,1)[1];

        double xsquare = Math.pow(x2-x1, 2);
        double ysquare = Math.pow(y2-y1, 2);
        double distance= Math.sqrt(xsquare+ysquare);
        double cosarg= Math.max(Math.abs(y2-y1),Math.abs(x2-x1));
        System.out.println("y1-y2= "+(y1-y2)+" x1-x2= "+(x1-x2));

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
            corners= detectAruco(rotated_image2);
            Mat detect2 = corners.get(0);
            double x12= detect2.get(0,0)[0];
            double y12= detect2.get(0,0)[1];
            double x22= detect2.get(0,1)[0];
            double y22= detect2.get(0,1)[1];
            double x32= detect2.get(0,2)[0];
            double y32= detect2.get(0,2)[1];
            double x42= detect2.get(0,3)[0];
            double y42= detect2.get(0,3)[1];
            double[] sec_dect= new double[]{x12,x22,x32,x42};
            System.out.println(GreatestandSecondGreatestDiff(sec_dect));
            System.out.println(GreatestandSecondGreatestDiff(first_dect));
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

            casse=caseFind(x1,y1,x2,y2,x3,y3,x4,y4);
            System.out.println(casse);
            double ratio= distance/5;
Mat cropImage=new Mat();
            System.out.println("Ratio: "+ratio);
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


        Imgcodecs.imwrite("output_with_boxes.png", undistortImg);
        String onnxPath = "C:\\Users\\vt\\runs\\detect\\yolov8n_custom4\\weights\\best.onnx";

    }

}