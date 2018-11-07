#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "face_tracker.h"


void detectAndDraw( cv::Mat& img, cv::CascadeClassifier& cascade, FaceTracker& faceTracker, const int& nFrame);

int main( int argc, const char** argv )
{
    // Will keep track of faces frame-to-frame
    FaceTracker faceTracker = FaceTracker::FaceTracker();
    cv::VideoCapture capture;
    cv::Mat frame, image;
    cv::CascadeClassifier cascade;
    cv::CommandLineParser parser(argc, argv,
        "{cascade|./haarcascade_frontalface_alt.xml|}"
    );

    std::string cascadeName = parser.get<std::string>("cascade");
    
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    
    if( !cascade.load( cascadeName ) )
    {
        std::cerr << "ERROR: Could not load classifier cascade" << std::endl;
        return -1;
    }
    
    if (!capture.open(0)) {
        std::cout << "Cannot open camera...";
        return 0;
    }

    int nFrame = 0;
    assert(capture.isOpened());

    std::cout << "Video capturing has been started ..." << std::endl;

    for(;;)
    {
        capture >> frame;
        if( frame.empty() )
            break;

        detectAndDraw(frame, cascade, faceTracker, nFrame);

        // Can quit at any time by pressing Q/q
        char c = (char)cv::waitKey(10);
        if( c == 27 || c == 'q' || c == 'Q' )
            break;

        nFrame += 1;
    }

    return 0;
}

// Copy the sub image into the image at the given ROI
void copySubImage(cv::Mat & img, cv::Mat & subImg, cv::Rect & roi) {
    assert(roi.width == subImg.cols);
    assert(roi.height == subImg.rows);
    const int offsetx = roi.x; // column offset
    const int offsety = roi.y; // row offset
    for(int i = 0; i < roi.height; i++) { // iterate rows
        for(int j = 0; j < roi.width; j++) { // iterate columns

            // cv::Point is (column/row) 
            cv::Vec4b colorSub = subImg.at<cv::Vec4b>(cv::Point(j, i));

            // If alpha channel is zero, skip pixel
            if (colorSub[3] != 0) {
                img.at<cv::Vec4b>(cv::Point(j+offsetx, i+offsety)) = colorSub;
            }
        }
    }

}

// Resize ROI to scale, truncating to image if extending beyond image bounds
cv::Rect resizeRoi(const cv::Rect & roi, const double & scale, const int & max_row, const int & max_col) {
    assert(scale > 0);

    int nwidth= int(float(roi.width) * scale);
    int nheight = int(float(roi.height) * scale); 
    int nx = roi.x - (nwidth - roi.width)/2;
    int ny = roi.y - (nheight - roi.height)/2;

    // If ROI is bigger than boundaries of image, truncate to fit
    if ( (nx + nwidth) > max_col) {
        nwidth = max_col - nx;
    }
    if ( (ny + nheight) > max_row) {
        nheight = max_row - ny;
    }
    if (nx < 0) {
        nwidth = nwidth + nx;
        nx = 0;
    }
    if (ny < 0) {
        nheight = nheight + ny;
        ny = 0;
    }

    assert((nx + nwidth) <= max_col);
    assert((ny + nheight) <= max_row);

    return cv::Rect(nx, ny, nwidth, nheight);
}

std::vector<cv::Rect> detectFacesInFrame( cv::Mat& img, cv::CascadeClassifier& cascade) {
    std::vector<cv::Rect> faces;
    cv::Mat gray, smallImg;
    cvtColor( img, gray, cv::COLOR_BGR2GRAY );
    double fx = 1;
    resize( gray, smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR_EXACT );
    equalizeHist( smallImg, smallImg );
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        |cv::CASCADE_SCALE_IMAGE,
        cv::Size(30, 30) );
    return faces;
}

void detectAndDraw( cv::Mat& img, cv::CascadeClassifier& cascade, FaceTracker& faceTracker, const int& nFrame)
{
    // Make incoming image RBGA, need to work with alpha channel
    cv::Mat imga;
    cvtColor(img, imga, cv::COLOR_RGB2RGBA);

    std::vector<cv::Rect> facesInFrame = detectFacesInFrame(img, cascade);

    bool debug = true;
 
    // draw detections on each image
    if (debug) {
        for (auto& r: facesInFrame) {
            rectangle( imga, cv::Point(cvRound(r.x), cvRound(r.y)),
                       cv::Point(cvRound((r.x + r.width-1)), cvRound((r.y + r.height-1))),
                       cv::Scalar(255,255,0), 3, 8, 0);
        }
    }


    faceTracker.addNewDetections(facesInFrame, nFrame);

    auto& facesReal = faceTracker.getFaces();

    for(auto& face : facesReal) {

        const Mask& mask = face.getMask();

        const cv::Rect& currentRoi = face.lastPosition;

        // Will get the ROI to modify in the original image, truncating
        // it to our sub-image if it extends past the boundary
        cv::Rect maskRoi = resizeRoi(currentRoi, mask.getScale(), imga.rows, imga.cols);

        // Want to make the mask slightly larger because the face recognition
        // algorithms makes smaller rectangles than the whole face
        cv::Mat maskResize;
        resize(mask.getImg(), maskResize, cv::Size(maskRoi.width, maskRoi.height), 0, 0, cv::INTER_AREA);

        // Now copy image into subImage 
        copySubImage(imga, maskResize, maskRoi);

    }
    cv::imshow( "result", imga );
    
    // to debug frame by frame
    //cv::waitKey(0);
}
