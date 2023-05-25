import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import math

hammer = {"true_pos": 0,
          "false_pos": 0,
          "false_neg": 0,
          "true_neg": 0}
pliers = {"true_pos": 0,
          "false_pos": 0,
          "false_neg": 0,
          "true_neg": 0}
screw = {"true_pos": 0,
          "false_pos": 0,
          "false_neg": 0,
          "true_neg": 0}
screwdriver = {"true_pos": 0,
          "false_pos": 0,
          "false_neg": 0,
          "true_neg": 0}
tape_measure = {"true_pos": 0,
          "false_pos": 0,
          "false_neg": 0,
          "true_neg": 0}
wrench = {"true_pos": 0,
          "false_pos": 0,
          "false_neg": 0,
          "true_neg": 0}


with open("/home/adrian/SDU/Project_in_Advanced_Robotics/Text-Based-Pick-and-Place/data/imgs/report/resultsYOLO/results_YOLO_cropped.txt", 'r') as f:
    images = 0
    total_items = 0
    correct = 0
    no_identified = 0
    miss_class = 0
    for line in f:
        images +=1
        if images == 90:
            break
        columns = line.split('\t')
        actual = []
        predicted = []
        for idx, column in enumerate(columns):
            if idx == 0:
                for word1 in column.split(','):
                    total_items += 1
                    if word1 != '\n':
                        actual.append(word1)
            if idx == 1:
                for word2 in column.split(','):
                    if word2 != '\n':
                        predicted.append(word2)
                
        print(actual)
        print(predicted)
        for element in actual:
            if element in predicted:
                correct +=1
                predicted.remove(element)
            else:
                no_identified += 1
        print(str(predicted) + '\n')
        for rest in predicted:
            miss_class += 1
           
print('Total number of images: \n' + str(images))
print('Total number of elements to be identified: \n' + str(total_items))
print('Total number of correct matches: \n' + str(correct))
print('Total number of no identified: \n' + str(no_identified))
print('Total number of missclasified: \n' + str(miss_class))

    
    