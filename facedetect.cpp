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
// TODO - add const correctness

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

        const Mat& getImg() const {
            return img;
        }

        double getScale() const {
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

    // Will be used to give new faces id's
    static int numFaces;

    public: 
        Mask mask;
        Rect lastPosition;
        double lastTimeSeen;
        int numDetections;
        int lastFrameSeen;
        int id;

        Face(Rect & detection, Mask& _mask, const int& nFrame) : mask(_mask) {
            lastPosition = detection;
            lastTimeSeen = (double)getTickCount();
            lastFrameSeen = nFrame;
            numDetections = 1;
            numFaces += 1;
            id = numFaces;
        }

        double timeUndetectedMs() const {
            return ((double)getTickCount() - this->lastTimeSeen)*1000 / getTickFrequency();
        }

        int undetectedFrames(const int& currentFrame) const {
            return currentFrame - lastFrameSeen;
        }
        
        void updateSeen(const Rect & detection, const int& nFrame) {
            lastPosition = detection;
            lastTimeSeen = (double)getTickCount();
            numDetections += 1;
            lastFrameSeen = nFrame;
            cout << "Updating face " << id << ": " << numDetections << " detections" << endl;
        }

        const Mask& getMask() const {
            return mask;
        }

};

int Face::numFaces = 0;


// Face tracker will keep track of the faces that have been seen and attempt to
// track them from frame to frame. There is a high false detection rate by the 
// face classifier, so we need to track stuff for a while before saying it's
// actually a real face
class FaceTracker {

    // Remove a face if it hasn't been seen for 10 frames
    static const int EXPIRE_FACE_FRAMES = 20;

    // Require being seen 10 times before considered real
    static const int DETECTIONS_REQUIRED_TO_BE_REAL = 8;

    // Limit total number of faces
    static const int MAX_FACES = 1;
    
    // Keep track of faces
    vector<Face> definiteFaces;
    vector<Face> potentialFaces;
    
    // Initialize the mask set for each face
    MaskVector masks;
    
    private:

        // Need to run stats on size and position similarity to check for a match
        bool matchesFace(const Rect & potential, Face & face, const int& nFrame) {
            const Rect& lastSeen = face.lastPosition;
            const int frameDiff = face.undetectedFrames(nFrame);

            // Percent moved in X direction relative to size of object
            double percxdiff = (double)abs(potential.x - lastSeen.x)/lastSeen.width;

            // Percent moved in Y direction relative to size of object
            double percydiff = (double)abs(potential.y - lastSeen.y)/lastSeen.height;

            // Ratio of width change
            double ratiow = (double)potential.width / lastSeen.width;

            // Ration of height change
            double ratioh = (double)potential.height / lastSeen.height;

            cout << endl;
            cout << "Face #" << face.id << " last seen " << frameDiff << " frames ago" << endl;
            cout << "Percent X: " << percxdiff << " Percent Y: " << percydiff << endl;
            cout << "Ratio width: " << ratiow << " Ratio height: " << ratioh << endl;

            bool match = true;

            // Be more strict if last seen more recently
            if (frameDiff <= 5) {
                if (ratiow < 0.9 || ratiow > 1.11 || ratioh < 0.9 || ratioh > 1.11) {
                    cout << "No match, width off" << endl;
                    match = false;
                } else if (percxdiff > 0.75 || percydiff > 0.75) {
                    // can't quickly move too far
                    cout << "No match, moved too far" << endl;
                    match = false;
                }
            } else {
                if (ratiow < 0.7 || ratiow > 1.43 || ratioh < 0.7 || ratioh > 1.43) {
                    cout << "No match, width off" << endl;
                    match = false;
                } else if (percxdiff > 1.5 || percydiff > 1.5) {
                    // can't quickly move too far
                    cout << "No match, moved too far" << endl;
                    match = false;
                }
            }
            return match;
        }

        // Check detections against a vector of faces for matches, and return
        // true if a match is found. 
        //
        // FIXME - future improvement to do a nearest neighbor check than return
        // first match
        bool matchesAnyFace(const Rect& detection, vector<Face>& faceVector, const int& nFrame, bool real = false) {
            bool foundMatch = false;
            for (auto & face : faceVector) {
                if (matchesFace(detection, face, nFrame)) {
                    cout << "Found a match for a " << (real ? "real" : "potential") << " face!" << endl;
                    face.updateSeen(detection, nFrame);
                    foundMatch = true;
                    continue;
                }
            }
            return foundMatch;
        }

        // Purge things that have not been seen in a while
        void purgeOldFaces(const int& nFrame) {
            definiteFaces.erase(std::remove_if(definiteFaces.begin(), 
                                               definiteFaces.end(), 
                                               [&](Face & i) { return i.undetectedFrames(nFrame) > EXPIRE_FACE_FRAMES; }), 
                                definiteFaces.end());
            potentialFaces.erase(std::remove_if(potentialFaces.begin(), 
                                               potentialFaces.end(), 
                                               [&](Face & i) { return i.undetectedFrames(nFrame) > EXPIRE_FACE_FRAMES; }), 
                                potentialFaces.end());
        }

        void upgradePotentialFaces() {
            // Copy faces over to definiteFaces if enough detections have been met
            std::copy_if(potentialFaces.begin(),
                         potentialFaces.end(),
                         std::back_inserter(definiteFaces), 
                         [](Face & i){return i.numDetections > DETECTIONS_REQUIRED_TO_BE_REAL;} );
            
            // And remove from potential
            potentialFaces.erase(std::remove_if(potentialFaces.begin(), 
                                                potentialFaces.end(), 
                                                [](Face & i) { return i.numDetections > DETECTIONS_REQUIRED_TO_BE_REAL; }), 
                                 potentialFaces.end());
        }
        
    public:

        FaceTracker() {
            masks = MaskVector();
            masks.emplace_back("imgs/hair2.png", 1.5);
            masks.emplace_back("imgs/guy3.png", 1.3);
            masks.emplace_back("imgs/glasses.png", 1.3);
        }


        // Take face detections from a frame and insert to the face tracker
        void addNewDetections(vector<Rect> detections, const int& nFrame) {

            cout << "Processing " << detections.size() << " detections" << endl << endl;

            for (auto & detection : detections) {

                // Check first for matching stuff we think is real
                if (matchesAnyFace(detection, definiteFaces, nFrame, true)) {
                    continue;

                }
                
                // Check first for matching stuff we think is potential
                if (matchesAnyFace(detection, potentialFaces, nFrame, false)) {
                   continue;
                }


                // If we are here, there is no matching face already existing, so 
                // it can be added to list of potentials
                potentialFaces.emplace_back(detection, masks.getNextMask(), nFrame);
                cout << "Adding new potential face." << endl;
            }

            // Now upgrade those with enough matches
            upgradePotentialFaces();

            // And purge the old ones
            purgeOldFaces(nFrame);

            cout << definiteFaces.size() << " real faces." << endl;
            cout << potentialFaces.size() << " potential faces." << endl;

        }

        // To draw what we have
        vector<Face>& getFaces() {
            return definiteFaces;
        }
};



void detectAndDraw( Mat& img, CascadeClassifier& cascade, MaskVector & masks, FaceTracker& faceTracker, const int& nFrame);
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

    int nFrame = 0;
    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;


        for(;;)
        {
            capture >> frame;
            if( frame.empty() )
                break;

            Mat frame1 = frame.clone();
            detectAndDraw( frame1, cascade, masks, faceTracker, nFrame);

            // ADD TOGGLE IDENTITY
            char c = (char)waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;

            nFrame += 1;
        }
    }
    else
    {
        cout << "Detecting face(s) in camera" << endl;
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, masks, faceTracker, nFrame);
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

vector<Rect> detectFacesInFrame( Mat& img, CascadeClassifier& cascade) {
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

void detectAndDraw( Mat& img, CascadeClassifier& cascade, MaskVector& masks, FaceTracker& faceTracker, const int& nFrame)
{
    // Make incoming image RBGA
    Mat imga;
    cvtColor(img, imga, COLOR_RGB2RGBA);

    vector<Rect> facesInFrame = detectFacesInFrame(img, cascade);

    bool debug = true;
 
    // draw detections on each image
    if (debug) {
        for (auto& r: facesInFrame) {
            rectangle( imga, Point(cvRound(r.x), cvRound(r.y)),
                       Point(cvRound((r.x + r.width-1)), cvRound((r.y + r.height-1))),
                       Scalar(255,255,0), 3, 8, 0);
        }
    }


/*
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect & face = faces[i];

        auto mask = masks.getNextMask();
*/
    faceTracker.addNewDetections(facesInFrame, nFrame);

    auto& facesReal = faceTracker.getFaces();

    for(auto& face : facesReal) {

        const Mask& mask = face.getMask();

        const Rect& currentRoi = face.lastPosition;

        // Will get the ROI to modify in the original image, truncating
        // it to our sub-image if it extends past the boundary
        Rect maskRoi = resizeRoi(currentRoi, mask.getScale(), imga.rows, imga.cols);

        // Want to make the mask slightly larger because the face recognition
        // algorithms makes smaller rectangles than the whole face
        Mat maskResize;
        resize(mask.getImg(), maskResize, Size(maskRoi.width, maskRoi.height), 0, 0, INTER_AREA);

        // Now copy image into subImage 
        copySubImage(imga, maskResize, maskRoi);

    }
    imshow( "result", imga );
    
    // to debug frame by frame
    //waitKey(0);
}
