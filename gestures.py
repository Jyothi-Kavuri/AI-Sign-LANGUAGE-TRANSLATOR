import cv2
import mediapipe as mp
import pyttsx3
import time

# Voice function
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate',150)
    engine.say(text)
    engine.runAndWait()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_gesture = ""
last_speak_time = 0

confidence = 0

while True:

    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    gesture = "NONE"

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_label = handedness.classification[0].label

            h, w, c = img.shape

            x_list = []
            y_list = []

            lm = hand_landmarks.landmark

            for id, lm_point in enumerate(lm):
                x, y = int(lm_point.x * w), int(lm_point.y * h)
                x_list.append(x)
                y_list.append(y)

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            cv2.rectangle(img,(xmin-20,ymin-20),(xmax+20,ymax+20),(0,255,0),2)

            fingers = []

            if lm[4].x > lm[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            if lm[8].y < lm[6].y:
                fingers.append(1)
            else:
                fingers.append(0)

            if lm[12].y < lm[10].y:
                fingers.append(1)
            else:
                fingers.append(0)

            if lm[16].y < lm[14].y:
                fingers.append(1)
            else:
                fingers.append(0)

            if lm[20].y < lm[18].y:
                fingers.append(1)
            else:
                fingers.append(0)

            # Gesture detection
            if fingers == [0,0,0,0,0]:
                gesture = "NO"
                confidence = 90

            elif fingers == [1,0,0,0,0]:
                gesture = "YES"
                confidence = 92

            elif fingers == [0,1,1,0,0]:
                gesture = "PEACE"
                confidence = 95

            elif fingers == [0,1,0,0,0]:
                gesture = "OK"
                confidence = 88

            elif fingers == [1,1,1,1,1]:
                gesture = "STOP"
                confidence = 93

            cv2.putText(img, hand_label,
                        (xmin-20,ymin-30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,0,0),
                        2)

    # Display panel
    cv2.rectangle(img,(0,0),(350,110),(0,0,0),-1)

    cv2.putText(img,
                "Gesture: "+gesture,
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.putText(img,
                "Confidence: "+str(confidence)+"%",
                (10,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2)

    # Voice control
    current_time = time.time()

    if gesture != "NONE":
        if gesture != last_gesture or current_time-last_speak_time > 2:
            speak(gesture)
            last_gesture = gesture
            last_speak_time = current_time

    cv2.imshow("AI Sign Language Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
