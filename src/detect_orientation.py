import numpy as np
import cv2
import os
import re
from math import atan2
import math

# to save the needed pictures: cv2.imwrite(data_path + "/orientation_pics/" + frame_path.replace(".npy", "") + ".jpg", frame)
data_path = os.getcwd() + "/fluently/data/"

angle_in_file = re.compile(r".*_(.*).jpg")


def draw_orientation(img, centre, eigenvectors, eigenvalues, angle):
    scale = 10  
    p1 = (centre[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0] * 100, centre[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0] * 100)
    p2 = (centre[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0] * 100, centre[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0] * 100)
    
    hypotenuse = np.sqrt((centre[1] - p1[1]) * (centre[1] - p1[1]) + (centre[0] - p1[0]) * (centre[0] - p1[0]))
    hypotenuse2 = np.sqrt((centre[1] - p2[1]) * (centre[1] - p2[1]) + (centre[0] - p2[0]) * (centre[0] - p2[0]))
    # Here we lengthen the arrow by a factor of scale
    p1[0] = centre[0] - scale * hypotenuse * np.cos(angle)
    q[1] = centre[1] - scale * hypotenuse * np.sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (0,255,0), 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * np.cos(angle + np.pi / 4)
    p[1] = q[1] + 9 * np.sin(angle + np.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (0,255,0), 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * np.cos(angle - np.pi / 4)
    p[1] = q[1] + 9 * np.np.sin(angle - np.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (0,255,0), 1, cv2.LINE_AA)


def vid_2_samples():
    """
    from the frames in the video directory let you select the images for the orientation algorithm
    """
    for frame_path in sorted(os.listdir(data_path + "video/")):
        frame = np.load(data_path + "video/" + frame_path)
        cv2.imshow("Frame", frame)
        cmd = cv2.waitKey(0)
        print(cmd)
        if cmd == 32:
            break
        elif cmd == 113:
            cv2.imwrite(data_path + "/orientation_pics/" + frame_path.replace(".npy", ".jpg"), frame)


def getOrientation(img):
    """
    return the angle of orientation of the object where 0 degree is the horizontal position and the +axis of rotation
    is clockwise and starting from the left of the circle

    Args:
        frame (cv2.Mat): The image of the ws thresholded and the background has been removed, on the image one single 
        object should be shown

    Returns:
        centre (tuple(float)): the x and y of the center of the object
        angle (float): the angle of the orientation of the object
        eigenvalues (tuple(float)): eigenvalues that will tell us the shape of the object 
    """
    contours, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 1e2 or 1e5 < area:
            continue

        sz = len(c)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i,0] = c[i,0,0]
            data_pts[i,1] = c[i,0,1]

        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

        # Store the center of the object
        centre = (int(mean[0,0]), int(mean[0,1]))

        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        cv2.circle(img, centre, 3, (255, 0, 255), 2)
        print(eigenvalues, eigenvectors)
        
        
    
        cv2.line(img, p1, p2, (255,255,255))
        return centre, angle, eigenvalues


for frame_path in sorted(os.listdir(data_path + "orientation_pics/")):
    frame = cv2.imread(data_path + "orientation_pics/" + frame_path, cv2.IMREAD_GRAYSCALE)
    centre, angle, eigenval = getOrientation(frame)

    # contours, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))

    # for i, c in enumerate(contours):
    #     # Calculate the area of each contour
    #     area = cv2.contourArea(c)
    #     # Ignore contours that are too small or too large
    #     if area < 1e2 or 1e5 < area:
    #         continue
    #     print("c: ", len(c))

    #     sz = len(c)
    #     data_pts = np.empty((sz, 2), dtype=np.float64)
    #     for i in range(data_pts.shape[0]):
    #         data_pts[i,0] = c[i,0,0]
    #         data_pts[i,1] = c[i,0,1]

    #     # Perform PCA analysis
    #     mean = np.empty((0))
    #     mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    #     # Store the center of the object
    #     centre = (int(mean[0,0]), int(mean[0,1]))

    #     angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians

    print("center: ", eigenval)
    print("Computed angle: ", angle * 180 / np.pi)
    actual_angle = np.double(angle_in_file.match(frame_path).group(1))
    print("actual angle:", actual_angle * 180 / np.pi)
    cv2.imshow("Video", frame)
    cv2.waitKey(0)
