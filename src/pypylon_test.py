from pypylon import pylon
import cv2
import random

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# demonstrate some feature access
new_width = camera.Width.GetValue() - camera.Width.GetInc()
# if new_width >= camera.Width.GetMin():
#     camera.Width.SetValue(new_width)

numberOfImagesToGrab = 1
camera.StartGrabbingMax(numberOfImagesToGrab)

index = 0

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data.
        print("SizeX: ", grabResult.Width)
        print("SizeY: ", grabResult.Height)
        img = grabResult.Array
        print("Gray value of first pixel: ", img[0, 0])
        cv2.imshow("img", img)
        cv2.imwrite("img/workspace/im%f.png" % random.randint(0, 100), img)
        cv2.waitKey()
        index += 1

    grabResult.Release()
camera.Close()
