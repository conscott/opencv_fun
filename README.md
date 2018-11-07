# Real-Time Multi-Face Tracking and Masking

Or something like a Snapchat filter....

## Description

The traditional OpenCV examples just provide face detections on a frame by frame basis, and are not correlated through time. Additionally they can have a high false-positive rate under different lighting conditions / video quailty, causing a lot of noise. The following program builds upon OpenCV's face classification tools to build a real-time face masking program that independently tracks multiple faces frame-to-frame using a primitive tracking algorithm.

### Details

Face detections on each frame are passed to the FaceTracker, which will keep an internal record of "potential" and "definite" faces. Incoming face detections are compared for both size and proximity similiarity to known faces to detect matches. If a potential face has enough detections, it will be upgraded to a "definite" face. Detected faces as given a "mask", or an overlay with some transparent alpha channel, that follows the face and scales with it as the face moves on scrren. Ultimately this works similiar to something like a Snapchat filter.

If a face has not been seen for some time, it will be dropped from the FaceTracker, however the Tracker can tolerate several frames worth of missed detections, and pick back up the face. 

### Requirements

- [OpenCV 3.4](https://opencv.org/releases.html) and its required dependencies

### Install
```
make
```

### Run
```
./facedetect
```

### File Walkthrough

* [facedetect.cpp](./facedetect.cpp): The main run loop that captures video from webcam and processes images
* [face.h](./face.h), [face.cpp](./face.cpp): The container for tracking face metadata
* [face_tracker.h](./face_tracker.h), [face_tracker.cpp](./face_tracker.cpp): Keeps track of faces and contains tracking algorithm logic
* [mask.h](./mask.h): Stores potential mask overlays for faces
