import cv2
import mediapipe as mp

print("Indian Sign Language Recognition using Quantum Machine Learning")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    cv2.imshow("ISL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
