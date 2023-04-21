import torch
import clip
from PIL import Image
from nltk.corpus import wordnet as wn
import nltk
import os
from nltk.chunk import RegexpParser
import spacy
from spacy import displacy
from time import time, sleep
from pypylon import pylon
import cv2
from ultralytics import YOLO
import cvzone
import math

model = YOLO("./data/tools_model.pt")
classNames = ['hammer', 'pliers', 'screwdriver', 'wrench']

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

numberOfImagesToGrab = 1
camera.StartGrabbingMax(numberOfImagesToGrab)


def classify(img):
    """_summary_

    Args:
        img (cv2.Mat): image to run classification on
        model (YOLO.model): model used for classification

    Returns:
        result (dict{label:bounding_box}): 
    """



while camera.IsGrabbing():
    img = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if img.GrabSucceeded():
       classify(img)

    img.Release()
camera.Close()

