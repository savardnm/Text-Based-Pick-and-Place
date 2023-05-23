from pypylon import pylon
import cv2
import random
import json
import numpy as np

x =  '{ "name":"basler_i40bp", "time": "4 November 2022", "mtx": [2669.4684193141475, 0, 1308.7439664763986, 0, 2669.6106447899438, 1035.4419708519022, 0, 0, 1], "dist": [-0.20357947, 0.1737404, -0.00051758, 0.00032546, 0.01914437], "h": 2048, "w": 2592}'
camera_info = json.loads(x)
mtx = np.reshape( np.array(camera_info["mtx"]), (3,3) )
dist = np.array(camera_info["dist"])
height = int(camera_info["h"])
width = int(camera_info["w"])
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(width,height),0, (width,height))

try:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    numberOfImagesToGrab = 10000
    camera.Open()
    camera.StartGrabbingMax(numberOfImagesToGrab)
    # Camera IP: 172.28.60.50
    # Must put into wired settings in ubuntu
except:
    print("NOT CONNECTED TO CAMERA! Using images instead silly")


def undistort_convert_frame(grabResult):
    """
    convert the grabresult from the camera and undistort it based on the script Frederik gave us

    Args:
        grabResult (pypylon.camera.RetrievResult): the frame grabbed from the camera

    Returns:
        img (cv2.Mat): the converted and undistorted image
    """
    img = grabResult.Array
    img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return img

index = 0

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        # Access the image data.
        img = undistort_convert_frame(grabResult)
        # cv2.imshow("img", img)
        cv2.imwrite("data/imgs/final/img_" + "{:04d}".format(index) + ".png", img)
    input("...")
    index += 1
    grabResult.Release()
camera.Close()
