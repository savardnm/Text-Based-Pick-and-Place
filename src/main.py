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
import numpy as np
from math import atan2

YOLO_model = YOLO("./data/tools_model.pt")
classNames = ['hammer', 'pliers', 'screwdriver', 'wrench']

nlp = spacy.load("en_core_web_sm")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

data_path = "./data/imgs/"

# camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
# camera.Open()
# numberOfImagesToGrab = 1
# camera.StartGrabbingMax(numberOfImagesToGrab)


def find_token(tree, deps):
    # Searches a tree for a given dep (word type)
    if tree['dependency'] in deps:
        return (True, tree)
    
    for child in tree['children']:
        found, sub_tree = find_token(child, deps)

        if found:
            return (True, sub_tree)
        
    return (False, None)


def build_tree(token):
    candidate = {'text': token.text, 'dependency': token.dep_, }

    # if token.n_lefts + token.n_rights > 0:
    candidate['children'] = [build_tree(child) for child in token.children]
    return candidate


def get_descriptors(token, descriptor_dependencies):
    descriptors = []
    for child in token['children']:
        if child['dependency'] in descriptor_dependencies:
            descriptors += [child['text']]
    return descriptors


def translate_query(sentence):
    doc = nlp(sentence)
    
    root = [token for token in doc if token.dep_ == 'ROOT'][0]
    tree = build_tree(root)

            
    success_pick, pick = find_token(tree, ["dobj"])
    success_place, place = find_token(tree, ["pobj"])

    if not (success_pick and success_place):
        print("transform query: failure")


    descriptor_dependencies = ['amod', 'compound']

    pick_descriptors = get_descriptors(pick, descriptor_dependencies)
    place_descriptors = get_descriptors(place, descriptor_dependencies)

    # print(pick['text'], "-->",place['text'])
    # print(pick['text'],": ", pick_descriptors)
    # print(place['text'],": ", place_descriptors)
    return pick, place, pick_descriptors, place_descriptors


def classify(img):
    """_summary_

    Args:
        img (cv2.Mat): image to run classification on
        model (YOLO.model): model used for classification

    Returns:
        result (ultralytics boxes class): the result of the classification
        (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/engine/results.py)
        int representing label: boxes.cls[0])
        tuple for bounding box boxes.xyxy 
        confidence: boxes.conf[0]
    """
    results = YOLO_model.predict(img)
    return results[0].boxes


def cut_snapshot_obj(img, top_left, bottom_right):
    return img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


def find_with_clip(candidate_list, pick, descriptors):
    logit_list = []
    for candidate in candidate_list:
        space_separated = [word + " " for word in descriptors] + [pick['text']]
        sentence = ''.join(space_separated)
        image = ???????
        text = clip.tokenize([sentence]).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text)
            
            logits_per_image, logits_per_text = clip_model(image, text)

            logit_list.append(logits_per_image)
            print(logit_list)

            # print(logits_per_image, logits_per_text)
            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    logit_tup = tuple(logit_list)

    logits_accross = torch.cat(logit_tup,1)

    print(logits_accross)

    probs = logits_accross.softmax(dim=-1).cpu().numpy()

    print("'" + sentence + "'" + " Likelihood:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

    return np.unravel_index(probs.argmax(), probs.shape)[0] 


def find_object_pose(frame):
    """
    return the angle of orientation of the object where 0 degree is the horizontal position and the +axis of rotation
    is clockwise and starting from the left of the circle

    Args:
        frame (cv2.Mat): The image of the ws thresholded and the background has been removed, on the image one single 
        object should be shown

    Returns:
        cntr (tuple(float)): the x and y of the center of the object
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
        center = (int(mean[0,0]), int(mean[0,1]))

        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians

        return center, angle, eigenvalues


# while camera.IsGrabbing():
while True:
    query = "move the large red wrench onto wooden paintbrush" #  input("Insert query...\n")
    if len(query) > 1: 
        pick, place, pick_des, place_des = translate_query(query)
        img = cv2.imread(data_path + "workbenches/5_jpg.rf.7f1bb1da7d89148e07c76acb830981c2.jpg") #  camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if img is not None: #if img.GrabSucceeded():
            objs_detected = classify(img)
            hit_list = []
            for obj in objs_detected:
                if classNames[int(obj.cls[0])] == pick['text']:
                    top_left = (int(obj.xyxy[0][0]), int(obj.xyxy[0][1]))
                    bottom_right = (int(obj.xyxy[0][2]), int(obj.xyxy[0][3]))
                    snap = cut_snapshot_obj(img, top_left, bottom_right)  
                    hit_list.append(snap) # save the index of all "wrench" if the query is asking for one and so on
            correct_index = find_with_clip(hit_list, pick, pick_des)
            print(correct_index)
            center, angle, eigen_val = find_object_pose(hit_list[correct_index])
            # TODO: grab with ur
    break
        # img.Release()
# camera.Close()

