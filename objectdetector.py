import numpy as np
import cv2
import scipy
from scipy.spatial import distance as dist
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

Known_distance = 30  # Inches
Known_width = 5.7  # Inches
thres = 0.5
nms_threshold = 0.2 
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)

font = cv2.FONT_HERSHEY_PLAIN
fonts = cv2.FONT_HERSHEY_COMPLEX
fonts2 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fonts3 = cv2.FONT_HERSHEY_COMPLEX_SMALL
fonts4 = cv2.FONT_HERSHEY_TRIPLEX

# Camera Object
cap = cv2.VideoCapture(0)  # Number According to Camera
face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Distance_level = 0
classNames = []
with open('coco.names', 'r') as f:
    classNames = f.read().splitlines()
print(classNames)
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output21.mp4', fourcc, 30.0, (640, 480))

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance

def face_data(image, CallOut, Distance_level):
    face_width = 0
    face_x, face_y = 0, 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        line_thickness = 2
        LLV = int(h * 0.12)
        cv2.line(image, (x, y + LLV), (x + w, y + LLV), GREEN, line_thickness)
        cv2.line(image, (x, y + h), (x + w, y + h), GREEN, line_thickness)
        cv2.line(image, (x, y + LLV), (x, y + LLV + LLV), GREEN, line_thickness)
        cv2.line(image, (x + w, y + LLV), (x + w, y + LLV + LLV), GREEN, line_thickness)
        cv2.line(image, (x, y + h), (x, y + h - LLV), GREEN, line_thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - LLV), GREEN, line_thickness)

        face_width = w
        face_center = []
        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y
        if Distance_level < 10:
            Distance_level = 10

        if CallOut == True:
            cv2.line(image, (x, y - 11), (x + 180, y - 11), ORANGE, 28)
            cv2.line(image, (x, y - 11), (x + 180, y - 11), YELLOW, 20)
            cv2.line(image, (x, y - 11), (x + Distance_level, y - 11), GREEN, 18)

    return face_width, faces, face_center_x, face_center_y

ref_image = cv2.imread("lena.png")
ref_image_face_width, _, _, _ = face_data(ref_image, False, Distance_level)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
print(Focal_length_found)

while True:
    _, frame = cap.read()
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    face_width_in_frame, Faces, FC_X, FC_Y = face_data(frame, True, Distance_level)
    if len(classIds) != 0:
        for i in indices:
            i = i
            box = bbox[i]
            confidence = str(round(confs[i], 2))
            color = Colors[classIds[i] - 1]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(frame, classNames[classIds[i] - 1] + " " + confidence, (x + 10, y + 20), font, 1, color, 2)

    for (face_x, face_y, face_w, face_h) in Faces:
        if face_width_in_frame != 0:
            Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
            Distance = round(Distance, 2)
            Distance_level = int(Distance)
            cv2.putText(frame, f"Distance {Distance} Inches", (face_x - 6, face_y - 6), fonts, 0.5, (BLACK), 2)

    if len(bbox) > 0:
        stack_x = []
        stack_y = []
        stack_x_print = []
        stack_y_print = []
        global D
        closest_object_name = ""

        for i in range(0, len(bbox)):
            x1 = bbox[i][0]
            y1 = bbox[i][1]
            x2 = bbox[i][0] + bbox[i][2]
            y2 = bbox[i][1] + bbox[i][3]

            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            stack_x.append(mid_x)
            stack_y.append(mid_y)
            stack_x_print.append(mid_x)
            stack_y_print.append(mid_y)

            frame = cv2.circle(frame, (mid_x, mid_y), 3, [0, 0, 255], -1)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)

        if len(bbox) == 2:
            D = int(dist.euclidean((stack_x.pop(), stack_y.pop()), (stack_x.pop(), stack_y.pop())))
            frame = cv2.line(frame, (stack_x_print.pop(), stack_y_print.pop()), (stack_x_print.pop(), stack_y_print.pop()), (255, 255, 255), 1)

            if D > 0 and D <= 40:
                # Determine the name of the object that's too close
                for i in range(len(bbox)):
                    x1, y1, w1, h1 = bbox[i]
                    object_name = classNames[classIds[i] - 1]
                    closest_object_name = object_name
                    break

                if closest_object_name:
                    speak(f"Warning! {closest_object_name} is very close.")

    cv2.imshow("Detection", frame)

    # Save Video
    out.write(frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
