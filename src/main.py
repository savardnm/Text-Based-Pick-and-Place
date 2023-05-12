import torch
import clip
from PIL import Image
import spacy
from time import time
from pypylon import pylon
import cv2
from ultralytics import YOLO
import numpy as np
from math import atan2
import time
import json

from nltk.corpus import wordnet as wn
from spacy import displacy
import nltk
import os
from nltk.chunk import RegexpParser

# ======== GENERAL SETTINGS ========
imgs_path = "./data/imgs/"

# ======== YOLO SETTINGS ========
YOLO_model = YOLO("./data/tools_model.pt")
classNames = ['wrench', 'pliers', 'screwdriver', 'hammer', 'tape measure', 'screw']

# ======== SPACY SETTINGS ========
nlp = spacy.load("en_core_web_sm")

# ======== CLIP SETTINGS ========
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# ======== BACKGROUND SETTINGS ========
background = cv2.imread(imgs_path + "background.png")
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# ======== PYPYLON SETTINGS ========
# camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
# camera.Open()
# numberOfImagesToGrab = 1
# camera.StartGrabbingMax(numberOfImagesToGrab)
# Camera IP: 172.28.60.50
# Must put into wired settings in ubuntu

# ======== UNDISTORTION SETTINGS ========
x =  '{ "name":"basler_i40bp", "time": "4 November 2022", "mtx": [2669.4684193141475, 0, 1308.7439664763986, 0, 2669.6106447899438, 1035.4419708519022, 0, 0, 1], "dist": [-0.20357947, 0.1737404, -0.00051758, 0.00032546, 0.01914437], "h": 2048, "w": 2592}'
camera_info = json.loads(x)
mtx = np.reshape( np.array(camera_info["mtx"]), (3,3) )
dist = np.array(camera_info["dist"])
height = int(camera_info["h"])
width = int(camera_info["w"])
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(width,height),0, (width,height))


def find_token(tree, deps):
    """
    Searches a tree for a given dep (word type)

    Args:
        tree (_type_): _description_
        deps (_type_): _description_

    Returns:
        _type_: _description_
    """
    if tree['dependency'] in deps:
        return (True, tree)
    
    for child in tree['children']:
        found, sub_tree = find_token(child, deps)

        if found:
            return (True, sub_tree)
        
    return (False, None)


def build_tree(token):
    """
    _summary_

    Args:
        token (_type_): _description_

    Returns:
        _type_: _description_
    """
    candidate = {'text': token.text, 'dependency': token.dep_, }

    # if token.n_lefts + token.n_rights > 0:
    candidate['children'] = [build_tree(child) for child in token.children]
    return candidate


def get_descriptors(token, descriptor_dependencies):
    """
    _summary_

    Args:
        token (_type_): _description_
        descriptor_dependencies (_type_): _description_

    Returns:
        _type_: _description_
    """
    descriptors = []
    for child in token['children']:
        if child['dependency'] in descriptor_dependencies:
            descriptors += [child['text']]
    return descriptors


def translate_query(sentence):
    """
    _summary_

    Args:
        sentence (_type_): _description_

    Returns:
        _type_: _description_
    """
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

    # #print(pick['text'], "-->",place['text'])
    # #print(pick['text'],": ", pick_descriptors)
    # #print(place['text'],": ", place_descriptors)
    return pick, place, pick_descriptors, place_descriptors


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


def classify(img):
    """
    classifiy the image creating bounding boxes with x,y position and label saying which object they are most likely to be

    Args:
        img (cv2.Mat): image to run classification on
        model (YOLO.model): model used for classification

    Returns:
        result (ultralytics boxes class): the result of the classification
        (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/engine/results.py)
        {int representing label: boxes.cls[0]), tuple for bounding box boxes.xyxy, confidence: boxes.conf[0]}
    """
    results = YOLO_model.predict(img)
    return results[0].boxes


def cut_snapshot_obj(img, top_left, bottom_right):
    """cut a snapshot out of img from top_left to bottom_right

    Args:
        img (cv2.Mat): the source image
        top_left (tuple(int, int)): x,y coordinate for the top left corner of the bounding box
        bottom_right (tuple(int, int)): x,y coordinate for the bottom right corner of the bounding box

    Returns:
        img (cv2.Mat): snapshot of the object
    """
    return img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


def create_candidates_list(pick, place, objs_detected):
    """_summary_

    Args:
        pick (_type_): _description_
        place (_type_): _description_
        objs_detected (_type_): _description_

    Returns:
        pick_hit_list (list(cv2.Mat)): the list of image that are candidates for pick  
        place_hit_list (list(cv2.Mat)): the list of image that are candidates for place
        pick_bg_list (list(cv2.Mat)): the list of background for each candidates for pick
        place_bg_list (list(cv2.Mat)): the list of background for each candidates for palce
        pick_coord_list (tuple(int)): the list of where the snap has the top left corner for each candidate, 
        place_coord_list (tuple(int)): we will use it to go back to global   
    """
    pick_hit_list = []
    place_hit_list = [] 
    pick_bg_list = []
    place_bg_list = []
    pick_coord_list = []
    place_coord_list = []
    for obj in objs_detected: # for each object detect we save it if it's either pick or place object
        #print(classNames[int(obj.cls[0])])
        top_left = (int(obj.xyxy[0][0]), int(obj.xyxy[0][1]))
        bottom_right = (int(obj.xyxy[0][2]), int(obj.xyxy[0][3]))
        if classNames[int(obj.cls[0])] == pick['text']:
            snap = cut_snapshot_obj(img, top_left, bottom_right)  
            bg_snap = cut_snapshot_obj(background, top_left, bottom_right) # to be optimized
            pick_bg_list.append(bg_snap)
            pick_hit_list.append(snap) # save the index of all "wrench" if the query is asking for one and so on
            pick_coord_list.append(top_left)
        if classNames[int(obj.cls[0])] == place['text']:
            snap = cut_snapshot_obj(img, top_left, bottom_right)  
            bg_snap = cut_snapshot_obj(background, top_left, bottom_right) # to be optimized
            place_hit_list.append(snap) # save the index of all "wrench" if the query is asking for one and so on
            place_bg_list.append(bg_snap)
            place_coord_list.append(top_left)
    return pick_hit_list, place_hit_list, pick_bg_list, place_bg_list, pick_coord_list, place_coord_list


def find_with_clip(candidate_list, pick, descriptors):
    """
    find the most matching image from a list of candidate images 
    [from 5 image of hammers we want to find which one is the one we re looking for]

    Args:
        candidate_list (list(cv2.Mat)): the list of candidate images
        pick (String): the name of the object we are looking for
        descriptors (list(String)): the descriptors for our object

    Returns:
        index (int): the index in the list of candidates that has the higher probability of it being the right image
    """
    logit_list = []
    for candidate in candidate_list:
        space_separated = [word + " " for word in descriptors] + [pick['text']]
        sentence = "A centered photo of " + ' '.join(space_separated)
        #print("query:", sentence)
        image = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
        #cv2.imshow("candidate image", candidate)
        #cv2.waitKey(0)
        image = Image.fromarray(image)
        image = preprocess(image).unsqueeze(0).to(device)
        text = clip.tokenize([sentence]).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text)
            
            logits_per_image, logits_per_text = clip_model(image, text)

            logit_list.append(logits_per_image)
            #print(logit_list)

    logit_tup = tuple(logit_list)

    logits_accross = torch.cat(logit_tup,1)

    # #print(logits_accross)

    #print("logits:", logits_accross)

    probs = logits_accross.softmax(dim=-1).cpu().numpy()

    # #print("'" + sentence + "'" + " Likelihood:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

    return probs.argmax()


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
    _, new_frame = cv2.threshold(new_frame, 30, 255, 0)
    return new_frame


def find_object_pose(frame):
    """
    return the angle of orientation of the object where 0 degree is the horizontal position and the +axis of rotation
    is clockwise and starting from the left of the circle

    Args:
        frame (cv2.Mat): The image of the ws thresholded and the background has been removed, on the image one single 
        object should be shown

    Returns:
        cntr (tuple(float)): the x and y of the center of the object
        angle (float): the angle of the orientation of the object in radians
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


def find_centre_in_global_img(local_centre, offset):
    """
    find the centre of the object in the whole image from the position of the centre in the snap
    and the position of the bounding box for the object

    Args:
        local_centre (tuple(int, int)): the position of the centre in the snapshot of th object
        offset (tuple(int, int)): the position of the top left corner of the bounding box of the object 

    Returns:
        global_centre (tuple(int, int)): the position of the centre in the whole image 
    """
    global_centre = [local_centre[0] + offset[0], local_centre[1] + offset[1]]
    return global_centre


def compute_world_pose(position, angle):
    """
    return the pose in the real world frame 

    Args:
        position ((tuple(int, int))): the position of the centre of the object in the frame of the camera
        angle (np.float): the angle in radians 

    Returns:
        pose (matrix): the pose in the real world of the object
    """
    pose = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
    return pose


# while camera.IsGrabbing():
while True:
    # query = "move the red screwdriver onto the orange screwdriver" 
    img = cv2.imread(imgs_path + "undistort_cropped/00031.png") # TEST
    #cv2.imshow("Original image" , img)
    #cv2.waitKey(0)
    query = input("Insert query...\n")
    start = time.time()
    if len(query) > 1: 
        pick, place, pick_des, place_des = translate_query(query)
        # grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if True: # grabResult.GrabSucceeded():
            # img = undistort_convert_frame(grabResult)
            objs_detected = classify(img)
            
            (pick_hit_list, place_hit_list, pick_bg_list, 
             place_bg_list, pick_coord_list, place_coord_list) = create_candidates_list(pick, place, objs_detected)
            
            pick_correct_index = find_with_clip(pick_hit_list, pick, pick_des)
            pick_correct_image = remove_bg(pick_hit_list[pick_correct_index], pick_bg_list[pick_correct_index])
            

            place_correct_index = find_with_clip(place_hit_list, place, place_des)
            place_correct_image = remove_bg(place_hit_list[place_correct_index], place_bg_list[place_correct_index])
            
            
            pick_local_centre, pick_angle, pick_eigen_val = find_object_pose(pick_correct_image)
            pick_centre = find_centre_in_global_img(pick_local_centre, pick_coord_list[pick_correct_index])
            
            place_local_centre, place_angle, place_eigen_val = find_object_pose(place_correct_image)
            place_centre = find_centre_in_global_img(place_local_centre, place_coord_list[place_correct_index])
            
            #print("Pick object angle: ", pick_angle * 180 / np.pi)
            #print("Place object angle: ", place_angle * 180 / np.pi)

            print("Execution time: ", time.time() - start, "[s]")

            cv2.imshow("Correct pick", pick_correct_image)
            cv2.waitKey(0)
            cv2.imshow("Correct place", place_correct_image)
            cv2.waitKey(0)
            
            cv2.destroyAllWindows()

        # grabResult.Release()
    # break
# camera.Close()

