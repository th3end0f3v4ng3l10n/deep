#!/bin/env python3
import cv2 as cv
import numpy as np
import lanes



cap = cv.VideoCapture('test.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame = cv.resize(frame,(1300,800))
    copy_img = np.copy(frame)
    
    try:
        frame = lanes.canny(frame)
        frame = lanes.mask(frame)
        lines = cv.HoughLinesP(frame,2, np.pi/180, 100, np.array([()]), minLineLength=20,maxLineGap= 5)
        averaged_lines = lanes.average_slope_intercept(frame,lines)
        line_image = lanes.display_lines(copy_img, averaged_lines)
        combo = cv.addWeighted(copy_img, 0.8, line_image,0.5,1)
        cv.imshow('frame', combo)
    except:
        pass
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
