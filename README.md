# Hand-gesture-mouse-control
This project allows you to control your computer mouse using hand gestures detected through your webcam.
install this


pip install opencv-python mediapipe pyautogui numpy

//code

import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

smoothening = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            index_tip = hand_landmarks.landmark[8]

            h, w, c = frame.shape
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)
    
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), cv2.FILLED)
        
            distance = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)

            curr_x = np.interp(index_x, (0, w), (0, screen_width))
            curr_y = np.interp(index_y, (0, h), (0, screen_height))
            
            curr_x = prev_x + (curr_x - prev_x) / smoothening
            curr_y = prev_y + (curr_y - prev_y) / smoothening
            
            pyautogui.moveTo(screen_width - curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
            
            if distance < 30:
                pyautogui.click()
                cv2.putText(frame, "Click", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Hand Gesture Mouse Control', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


//Features
Mouse Movement: Index finger controls the cursor position

Left Click: Pinch thumb and index finger together

Right Click: Pinch middle and index finger together

Scroll: Pinch all three fingers (thumb, index, middle) together and move up/down

Smooth Cursor Movement: Implemented to reduce jitter
