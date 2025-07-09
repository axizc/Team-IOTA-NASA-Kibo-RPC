package jp.jaxa.iss.kibo.rpc.sampleapk;
import java.util.ArrayList;
import java.util.List;

import org.opencv.aruco.Aruco;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.aruco.Dictionary;
import org.opencv.imgcodecs.Imgcodecs;

public class CropImage {

    public static void main(String[] args) {
        Mat image = Imgcodecs.imread("C:\\Users\\vt\\Downloads\\img3.png");
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        double[] NAVCAM_CAMERA_MATRIX_SIMULATION = new double[]{523.10575D, 0.0D, 635.434258D, 0.0D, 534.765913D, 500.335102D, 0.0D, 0.0D, 1.0D};
        cameraMatrix.put(0, 0, NAVCAM_CAMERA_MATRIX_SIMULATION);
        Mat cameraCoefficients = new Mat(1,5,CvType.CV_64F);
        double[] NAVCAM_DISTORTION_COEFFICIENTS_SIMULATION = new double[]{-0.164787D, 0.020375D, -0.001572D, -3.69E-4D, 0.0D};

        cameraCoefficients.put(0,0,NAVCAM_DISTORTION_COEFFICIENTS_SIMULATION );
        Mat undistortImg = new Mat();
        Calib3d.undistort(image, undistortImg, cameraMatrix, cameraCoefficients);

        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat markerIds = new Mat();
        Aruco.detectMarkers(undistortImg, dictionary, corners, markerIds);

        System.out.println(corners);

    }

}