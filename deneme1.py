#!/usr/bin/env python

#import urllib
#import rospy
import cv2
import sys
import numpy as np
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge, CvBridgeError 
import time

#url='http://192.168.0.12:8080/shot.jpg'

#def image_publisher():
    #pub = rospy.Publisher('/image', Image, queue_size=100)
    #pub = rospy.Publisher('/image', Image, queue_size=100)
    #rospy.init_node('image_publisher', anonymous=True)
    #rate = rospy.Rate(10) # not sure this is necessary
    #bridge = CvBridge()

    #cap = cv2.VideoCapture(0)
    #print "Correctly opened resource, starting to show feed."
    #rval, frame = cap.read()
    #while rval:

#cv2.imshow("Stream: ", frame)
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)

img = cv2.imread("aaa.jpg",cv2.IMREAD_GRAYSCALE)
sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img,None)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)





while True:
    #imgResp=urllib.urlopen(url)
    #imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    #frame=cv2.imdecode(imgNp,-1)
    _,frame=cap.read()
    grayframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #train image


    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)

    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    good_points = []
    for m, n in matches:
        if m.distance <  0.4*n.distance:
            good_points.append(m)
    
    if len(good_points) >5: 
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Perspective transform
        h, w = img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        cv2.imshow("Homography", homography)
    else:
        cv2.imshow("homograpy",grayframe)
    #cv2.imshow("image",img)
    #cv2.imshow("frame",grayframe)
    img3=cv2.drawMatches(img,kp_image,grayframe,kp_grayframe,good_points,grayframe)
    cv2.imshow("img3",img3)
    key= cv2.waitKey(1)

    
   

cv2.destroyAllWindows()
