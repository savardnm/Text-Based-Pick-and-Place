import numpy as np
import cv2
import os
import re
from math import atan2
import math

# to save the needed pictures: cv2.imwrite(data_path + "/orientation_pics/" + frame_path.replace(".npy", "") + ".jpg", frame)
data_path = os.getcwd() + "/data/imgs/"

angle_in_file = re.compile(r".*_(.*).jpg")

background = cv2.imread(data_path + "cropped_object_bg.png")
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = np.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * np.cos(angle)
    q[1] = p[1] - scale * hypotenuse * np.sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * np.cos(angle + np.pi / 4)
    p[1] = q[1] + 9 * np.sin(angle + np.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)
    p[0] = q[0] + 9 * np.cos(angle - np.pi / 4)
    p[1] = q[1] + 9 * np.sin(angle - np.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)


def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 0, 255), 0.6)
    drawAxis(img, cntr, p2, (0, 255, 0), 0.6)
    print("eigenvectors, ",eigenvectors)
    print("eigenvalues, ", eigenvalues)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return angle


def remove_bg(frame, background):
    """
    remove the background from the image and tresholded it to make it binary

    Args:
        frame (cv2.Mat): the snapshot of the object on which we have to remove the background
        background (_type_): the background of the snapshot 

    Returns:
        new_frame (cv2.Mat): the snapshot with his background removed and tresholded
    """
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_frame = cv2.absdiff(new_frame, background)
    _, new_frame = cv2.threshold(new_frame, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # _, new_frame = cv2.threshold(new_frame, 30, 255, 0)
    return new_frame


# frame = cv2.imread(data_path + "frame001_-4.900643900102279e-08.jpg", cv2.IMREAD_GRAYSCALE)
src = cv2.imread("./data/imgs/cropped_object.png")
gray = remove_bg(src, background)

_, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
areas = []
for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv2.contourArea(c)
    # Ignore contours that are too small or too large
    # if area < 1e2 or 1e5 < area:
    #     continue
    areas.append(area)
    # Draw each contour only for visualisation purposes
    # tmp = src.copy()
    # cv2.drawContours(tmp, c, -1, (0, 0, 255), 2)
    # cv2.imwrite("tmp" + str(area) + " " + str(i), tmp)
    # cv2.waitKey(0)    

right_contour = contours[np.argmax(areas)]    
# cv2.drawContours(src, right_contour, -1, (0, 0, 255), 2)
# Find the orientation of each shape
tmp = gray.copy()

output_img = cv2.merge([tmp, tmp, tmp])

angle = getOrientation(right_contour, output_img)
cv2.imwrite('data/imgs/orientation_output.png', output_img)

cv2.imshow("Output", output_img)
cv2.waitKey()
