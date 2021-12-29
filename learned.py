import cv2
import numpy as np
import math

def round_half_up(n, decimals=0):
    # https://realpython.com/python-rounding/#rounding-half-away-from-zero
    multiplier = 10 ** decimals
    return int(math.floor(n*multiplier + 0.5) / multiplier)



#######   training part    ############### 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE, responses)

############################# testing part  #########################

im = cv2.imread('learned.png')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) 
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
cv2.imshow('im',im)
cv2.imshow('out',out)
print (Matrix)
while (True):

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break


