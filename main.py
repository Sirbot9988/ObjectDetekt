# Imports
import numpy as np 
import cv2
import argparse
import imutils
from gtts import gTTS
from io import BytesIO
from pygame import mixer
import os
import pyglet
# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"mil": cv2.TrackerMIL_create,
}
if not args.get("tracker", False):
    tracker = OPENCV_OBJECT_TRACKERS["kcf"]
else: 
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize bounding box
initBB = None
boxLeftOrRight = None
lastBoxLorR = None
# Capturing Webcam
cap = cv2.VideoCapture(0)

def TextToSpeech(text):
    tts = gTTS(text=text, lang='en')
    filename = './temp.mp3'
    tts.save(filename)

    music = pyglet.media.load(filename, streaming=False)
    music.play()

    os.remove(filename) 


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Resize

    frame = imutils.resize(frame, width=500)
    (H,W) = frame.shape[:2] 
    mixer.init()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if initBB is not None: 
        (success, box) = tracker.update(frame)
        # Faces
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces: 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if success: 
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if W/2 < x: 
                boxLeftOrRight = "right"
            else: 
                boxLeftOrRight = "left"
            if lastBoxLorR is not boxLeftOrRight:
                TextToSpeech(boxLeftOrRight)
            

            lastBoxLorR = boxLeftOrRight
        info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
            ("Coordinates", str(x) + ", " + str(y) if success else "None"),
            ("Direction", boxLeftOrRight if success else "None")
		]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()