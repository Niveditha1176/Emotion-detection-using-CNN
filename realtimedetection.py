import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------------- LOAD MODEL ----------------
model = load_model("final_emotion_model.keras")

# ---------------- FACE DETECTOR ----------------
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# ---------------- LABELS ----------------
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(image):
    image = np.array(image)
    image = image.reshape(1, 48, 48, 1)
    image = image / 255.0
    return image

# ---------------- WEBCAM ----------------
webcam = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))

        img = extract_features(face_img)

        prediction = model.predict(img, verbose=0)
        prediction_label = labels[np.argmax(prediction)]

        # draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # put text
        cv2.putText(
            frame,
            prediction_label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Emotion Detection", frame)

    # press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------------- CLEANUP ----------------
webcam.release()
cv2.destroyAllWindows()