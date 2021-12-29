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



# training part
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE, responses)

#

def getMatrix(model, im):

	out = np.zeros(im.shape,np.uint8)
	gray = im # cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) 
	# thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
	thresh = gray
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	xbin = 400/6.0
	ybin = 460/10.0
	Matrix = [[0 for x in range(6)] for y in range(10)] 


	for cnt in contours:
		if cv2.contourArea(cnt)>50:
			[x,y,w,h] = cv2.boundingRect(cnt)
			if  h>27:
				cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
				roi = thresh[y:y+h,x:x+w]
				roismall = cv2.resize(roi,(10,10))
				roismall = roismall.reshape((1,100))
				roismall = np.float32(roismall)
				retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
				integer = int((results[0][0]))
				string = str(integer)
				if (x==0 and y == 0) :
					continue
				cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
				#if (string == "2"):
				print (round(x/xbin), round(y/ybin), string)
				Matrix[round(y/ybin)][round(x/xbin)] = integer
	# cv2.imshow('im',im)
	# cv2.imshow('out',out)
	return Matrix


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
	
	print(getMatrix(model, frame))
	
	## 

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()


