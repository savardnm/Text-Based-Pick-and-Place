from pypylon import pylon

import cv2
import numpy as np
import json

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

numberOfImagesToGrab = 100
camera.StartGrabbingMax(numberOfImagesToGrab)

x =  '{ "name":"basler_i40bp", "time": "4 November 2022", "mtx": [2669.4684193141475, 0, 1308.7439664763986, 0, 2669.6106447899438, 1035.4419708519022, 0, 0, 1], "dist": [-0.20357947, 0.1737404, -0.00051758, 0.00032546, 0.01914437], "h": 2048, "w": 2592}'


camera_info = json.loads(x)
 
mtx = np.reshape( np.array(camera_info["mtx"]), (3,3) )
dist = np.array(camera_info["dist"])
height = int(camera_info["h"])
width = int(camera_info["w"])
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(width,height),0, (width,height))



while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    

    if grabResult.GrabSucceeded():
        # Access the image data.
        print("SizeX: ", grabResult.Width)
        print("SizeY: ", grabResult.Height)
        img = grabResult.Array
        img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        
        
        img = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        print("shape: ", img.shape)

        cv2.imshow("img", img)
        cv2.waitKey(50)

        print("Gray value of first pixel: ", img[0, 0])

    grabResult.Release()
camera.Close()
