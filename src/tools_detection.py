#In this code we are no longer using pretrained models in our project.
# The goal then will be create custom trained model with our own dataset


from ultralytics import YOLO
import cv2
import cvzone
import math

#cap = cv2.VideoCapture(0)  # For Webcam
#cap.set(3, 1280)
#cap.set(4, 720)
#cap = cv2.VideoCapture("../Videos/Tools.mp4")  # For Video

model = YOLO("./data/tools_model.pt")
classNames = ['wrench', 'pliers', 'screwdriver', 'hammer', 'tape measure', 'screw']
# file = open("/home/adrian/SDU/Project_in_Advanced_Robotics/Text-Based-Pick-and-Place/data/imgs/report/resultsYOLO/results_YOLO_cropped.txt", "w")
# file.write("Objects actually present\t" + "Objects detected\n")
# file.close()

for i in range(10,122):
    
    try:
        img = cv2.imread("/home/adrian/SDU/Project_in_Advanced_Robotics/Text-Based-Pick-and-Place/data/imgs/final/img_"+ "{:04d}".format(i) + ".png")  # For Image
        # img = img_[474:1850, 610:2000]
        # cv2.imshow("Image", cropped_image)
        # cv2.waitKey(50)
    except:
        continue
    
    # cv2.namedWindow("InputImg", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    # imS = cv2.resize(img, (960, 540))                # Resize image
    # cv2.imshow("InputImg", imS)  
    # cv2.waitKey(0)
    
    myColor = (0, 0, 255)

    #to run it on a video: - uncoment 'while True:' and indent everythign below it.
    #                      - uncoment success, img = cap.read()

    #while True:
    #success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        #print(r)
        boxes = r.boxes
        classes_detected = []
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            classes_detected.append(currentClass)
            print(currentClass)
            # if conf>0.9:
            if currentClass =='pliers': # to create different color for the boxes can be done in this way
                myColor = (0, 0,255)
            elif currentClass =='Hammer' or currentClass =='Screw Driver' or currentClass == 'Wrench':
                myColor =(0,255,0)
            else:
                myColor = (255, 0, 0)

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                            (max(0, x1), max(35, y1-8)), scale=1, thickness=1,colorB=myColor,
                            colorT=(255,255,255),colorR=myColor, offset=5)
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    imS = cv2.resize(img, (720, 1028))
    cv2.imshow("Image", imS)
    cv2.waitKey(50)
    
    # objects_image = input("Objects actually present in the imput image: \n")
    
    # file = open("/home/adrian/SDU/Project_in_Advanced_Robotics/Text-Based-Pick-and-Place/data/imgs/report/resultsYOLO/results_YOLO_cropped.txt", "a")
    # file.write(objects_image + "\t")
    # for item in classes_detected:
    #     file.write(str(item) + " ")
    # file.write("\n")
    # file.close()
    
    cv2.imwrite("/home/adrian/SDU/Project_in_Advanced_Robotics/Text-Based-Pick-and-Place/data/imgs/report/resultsYOLO/im_00000" + str(i) + ".png", img)
    
    #cv2.waitKey(0) #In image set 0, in video set 1 inside of the parenthesys
    # cv2.imwrite("./data/imgs/detected_workbench.jpg", img)