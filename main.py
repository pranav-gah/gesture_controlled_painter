import mediapipe as mp
import numpy as np 
import cv2 as cv
# from tkinter import *


finger_distance = 100
prev_x, prev_y = None, None
canvas = None
mode = None
col=input(" Enter the color you want of pen choose between Blue , Green , Red =  ")
if col=="blue":
    print("You choosed blue")
    pen_colour=[255,0,0]
elif col=="red":
    print("You choosed red")
    pen_colour=[0,0,255]
elif col=="green":
    print("You choosed green")
    pen_colour=[0,255,0]
else:
   print("Either you choosed some extrs color or white , so white color is assigned for pen ")
   pen_colour=[255,255,255]


# LOADING MEDIAPIPE -->


mp_hands = mp.solutions.hands #Loading hand detection module into mp_hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def distance( thumb,finger,mode):
    dist=np.linalg.norm(np.array(thumb) - np.array(finger))
    if(dist > finger_distance):
        mode = "pen"
    elif 14 < dist < 30:
        mode = "eraser"
        # print("helo")
    else: mode = None
    return dist , mode


cap=cv.VideoCapture(0)
while True:
    tip_x_thumb = 0
    tip_y_thumb = 0
    tip_x_finger = 0
    tip_y_finger = 0
    ret,frame=cap.read()
    if not ret :
        print("not getting frames")
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    h,w,_=frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            thumb = hand_landmarks.landmark[4]
            finger = hand_landmarks.landmark[8]
            tip_x_thumb = int(thumb.x * w)
            tip_y_thumb = int(thumb.y * h)
            tip_x_finger = int(finger.x * w)
            tip_y_finger = int(finger.y * h)
            

    d,mode=distance([tip_x_thumb,tip_y_thumb],[tip_x_finger,tip_y_finger],mode)

    if mode== "pen":
        if prev_x is not None and prev_y is not None:
            cv.line(
                canvas,
                (prev_x, prev_y),
                (tip_x_finger, tip_y_finger),
                (pen_colour), 
                2              
            )
        prev_x, prev_y = tip_x_finger, tip_y_finger

    if mode== "eraser":
        if prev_x is not None and prev_y is not None:
            cv.line(
                canvas,
                (prev_x, prev_y),
                (tip_x_finger, tip_y_finger),
                (0, 0, 0), 
                20           
            )
        prev_x, prev_y = tip_x_finger, tip_y_finger
    

    frame_to_show=frame.copy()
    frame_to_show=cv.putText(frame_to_show,f'draw = {mode} and Dist = {d} ',(20,60),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
    cv.circle(frame_to_show, (tip_x_thumb, tip_y_thumb), 8, (0, 255, 0), -1)
    cv.circle(frame_to_show, (tip_x_finger, tip_y_finger), 8, (0, 255, 0), -1)
    cv.line(frame_to_show, (tip_x_thumb, tip_y_thumb), (tip_x_finger, tip_y_finger), (255, 0, 0), 2)


    output = cv.add(frame_to_show, canvas)
    cv.imshow('webcam', output)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
