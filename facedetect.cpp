#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

// TODO - remove these
using namespace std;
using namespace cv;

// TODO - use initializer list for constructors
// TODO - return by reference
// TODO - don't forget auto

// Class to store some images we are going to import
class Mask
{

    private:
        string filename;
        double scale;
        Mat img;

    public:

        // Need to provide args, no default constructor
        Mask() = delete;

        Mask(string _filename, double _scale) {
            filename = _filename;
            scale = _scale;
            img = imread(filename, IMREAD_UNCHANGED);
        }

        Mat getImg() {
            return img;
        }

        double getScale() {
            return scale;
        }
};

// Subclass vector
class MaskVector : public vector<Mask> {

    private:
        int nextImg;

    public:

        // TODO -return reference?
        Mask& getNextMask() {
            assert(!this->empty());
            nextImg += 1;
            //return this->operator[](rand()%(this->size())).getImg();
            return this->operator[](nextImg%(this->size()));
        }
};


// A face we are tracking in the image
class Face {

    public: 
        Mask mask;
        Rect lastPosition;
        double lastTimeSeen;
        int numDetections;

    Face(Rect & detection, Mask& _mask) : mask(_mask) {
        lastPosition = detection;
        lastTimeSeen = (double)getTickCount();
        numDetections = 1;
    }

    double timeUndetectedMs() {
        return ((double)getTickCount() - this->lastTimeSeen)*1000 / getTickFrequency();
    }

};


// Face tracker will keep track of the faces that have been seen and attempt to
// track them from frame to frame. There is a high false detection rate by the 
// face classifier, so we need to track stuff for a while before saying it's
// actually a real face
class FaceTracker {

    // Remove a face if it hasn't been seen for 1000 ms
    static const int EXPIRE_FACE_MS = 1000;

    // Require being seen 10 times before considered real
    static const int DETECTIONS_REQUIRED_TO_BE_REAL = 10;

    // Limit total number of faces
    static const int MAX_FACES = 1;
    
    // Keep track of faces
    vector<Face> definiteFaces;
    vector<Face> potentialFaces;
    
    // Initialize the mask set for each face
    MaskVector masks;
    
    private:

        bool matchesFace(Rect & potential, Face & real) {
            return true;
        }

        void updateSeen(Face & face, Rect & detection) {
            face.lastPosition = detection;
            face.lastTimeSeen = (double)getTickCount();
            face.numDetections += 1;
        }
       

        // Purge things that have not been seen in a while
        void purgeOldFaces() {
            definiteFaces.erase(std::remove_if(definiteFaces.begin(), 
                                               definiteFaces.end(), 
                                               [](Face & i) { return i.timeUndetectedMs() > EXPIRE_FACE_MS; }), 
                                definiteFaces.end());
            potentialFaces.erase(std::remove_if(potentialFaces.begin(), 
                                               potentialFaces.end(), 
                                               [](Face & i) { return i.timeUndetectedMs() > EXPIRE_FACE_MS; }), 
                                potentialFaces.end());
        }

        void upgradePotentialFaces() {
            // Copy faces over to definiteFaces if enough detections have been met
            std::copy_if(potentialFaces.begin(),
                         potentialFaces.end(),
                         std::back_inserter(definiteFaces), 
                         [](auto & i){return i.numDetections > DETECTIONS_REQUIRED_TO_BE_REAL;} );
            
            // And remove from potential
            potentialFaces.erase(std::remove_if(potentialFaces.begin(), 
                                                potentialFaces.end(), 
                                                [](Face & i) { return i.numDetections > DETECTIONS_REQUIRED_TO_BE_REAL; }), 
                                 potentialFaces.end());
        }
        
    public:

        FaceTracker() {
            MaskVector masks = MaskVector();
            masks.emplace_back("imgs/hair2.png", 1.5);
            masks.emplace_back("imgs/guy3.png", 1.3);
            masks.emplace_back("imgs/glasses.png", 1.3);
        }


        // Take face detections from a frame and insert to the face tracker
        void addNewDetections(vector<Rect> detections) {

            for (auto & detection : detections) {

                // Check first for matching stuff we think is real
                for (auto & faceReal : definiteFaces) {
                    if (matchesFace(detection, faceReal)) {
                        cout << "Found a match for real face!";
                        updateSeen(faceReal, detection);
                        continue;
                    }
                }


                // Then check things we think might be real
                for (auto & facePot : potentialFaces) {
                    if (matchesFace(detection, facePot)) {
                        cout << "Found a match for potential face!";
                        updateSeen(facePot, detection);
                        continue;
                    }
                }

                // If we are here, there is no matching face already existing, so 
                // it can be added to list of potentials
                potentialFaces.emplace_back(detection, masks.getNextMask());
            }

            // Now upgrade those with enough matches
            upgradePotentialFaces();

            // And purge the old ones
            purgeOldFaces();
        }

};



void detectAndDraw( Mat& img, CascadeClassifier& cascade, MaskVector & masks, FaceTracker& faceTracker);
string cascadeName;


int main( int argc, const char** argv )
{
    //imshow("guyMask", guyMask);
    
    // Initialize our mask images
    MaskVector masks = MaskVector();
    //masks.emplace_back("imgs/hair2.png", 1.5);
    //masks.emplace_back("imgs/guy3.png", 1.3);
    masks.emplace_back("imgs/glasses.png", 1.0);

    // Will keep track of faces frame-to-frame
    FaceTracker faceTracker = FaceTracker();

    // TODO - clean this up
    VideoCapture capture;
    Mat frame, image;
    CascadeClassifier cascade;
    cv::CommandLineParser parser(argc, argv,
        "{cascade|./haarcascade_frontalface_alt.xml|}"
    );
    cascadeName = parser.get<string>("cascade");
    
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }
    
    if (!capture.open(0)) {
        cout << "Cannot open camera...";
        return 0;
    }

    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;

        for(;;)
        {
            capture >> frame;
            if( frame.empty() )
                break;

            Mat frame1 = frame.clone();
            detectAndDraw( frame1, cascade, masks, faceTracker);

            // ADD TOGGLE IDENTITY
            char c = (char)waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }
    }
    else
    {
        cout << "Detecting face(s) in camera" << endl;
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, masks, faceTracker);
            waitKey(0);
        }
    }

    return 0;
}

// Copy the sub image into the image at the given ROI
void copySubImage(Mat & img, Mat & subImg, Rect & roi) {
    assert(roi.width == subImg.cols);
    assert(roi.height == subImg.rows);
    const int offsetx = roi.x; // column offset
    const int offsety = roi.y; // row offset
    for(int i = 0; i < roi.height; i++) { // iterate rows
        for(int j = 0; j < roi.width; j++) { // iterate columns

            // Point is (column/row) 
            Vec4b colorSub = subImg.at<Vec4b>(Point(j, i));

            // If alpha channel is zero, skip pixel
            if (colorSub[3] != 0) {
                img.at<Vec4b>(Point(j+offsetx, i+offsety)) = colorSub;
            }
        }
    }

}

// Resize ROI to scale, truncating to image if extending beyond image bounds
Rect resizeRoi(const Rect & roi, const double & scale, const int & max_row, const int & max_col) {
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

    return Rect(nx, ny, nwidth, nheight);
}

vector<Rect> detectFaces( Mat& img, CascadeClassifier& cascade) {
    vector<Rect> faces;
    Mat gray, smallImg;
    cvtColor( img, gray, COLOR_BGR2GRAY );
    double fx = 1;
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT );
    equalizeHist( smallImg, smallImg );
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    return faces;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade, MaskVector& masks, FaceTracker& faceTracker)
{
    // Make incoming image RBGA
    Mat imga;
    cvtColor(img, imga, COLOR_RGB2RGBA);

    vector<Rect> faces = detectFaces(img, cascade);

    // Try out the face tracker
    //faceTracker.addNewDetections(faces);

    //printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect & face = faces[i];

        auto mask = masks.getNextMask();
        
        // Will get the ROI to modify in the original image, truncating
        // it to our sub-image if it extends past the boundary
        Rect faceRoi = resizeRoi(face, mask.getScale(), imga.rows, imga.cols);

        // Want to make the mask slightly larger because the face recognition
        // algorithms makes smaller rectangles than the whole face
        Mat maskResize;
        resize(mask.getImg(), maskResize, Size(faceRoi.width, faceRoi.height), 0, 0, INTER_AREA);

        // Now copy image into subImage 
        copySubImage(imga, maskResize, faceRoi);

    }
    imshow( "result", imga );
}
