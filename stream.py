# import the necessary packages
from imutils.video import VideoStream
import datetime
import argparse
import imutils
import time
import cv2

from mss import mss
from PIL import Image
import numpy as np


 

bounding_box = {'top': 300, 
    'left': 20, 
	'width': 600, 'height': 700}

sct = mss()
w = 400

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# frame = vs.read()

	original = sct.grab(bounding_box)
	frame = np.array(original)
	frame = imutils.resize(frame, width=w)

	#  center = (w // 2, h // 2)
	# M = cv2.getRotationMatrix2D(center, 180, 1.0)

	frame = cv2.GaussianBlur(frame, (3, 3), 0)
	(thresh,frame) = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)	
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	(thresh,frame) = cv2.threshold(frame, 254, 255, cv2.THRESH_BINARY)	
	
	# frame = cv2.Canny(frame, 30, 30)
	
	# frame = cv2.warpAffine(frame, M, (w, h))
	# draw the timestamp on the frame
	# timestamp = datetime.datetime.now()
	# ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	#cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		# 0.35, (0, 0, 255), 1)
	# show the frame
	frame = frame[5:600, 10:700]
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()


