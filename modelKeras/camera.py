import cv2
from model import FaceExpressionReconModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = FaceExpressionReconModel(path="FaceExpressRecon.h5")


class Camera:

    def __init__(self):
        self.videoCapture = cv2.VideoCapture(0)

    def __del__(self):
        self.videoCapture.release()

    def get_frame(self):
        _, frame = self.videoCapture.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = facec.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray_frame[y: y + h, x: x + w]

            face_resized = cv2.resize(face, (48, 48))

            pred = model.predictEmotion(face_resized[np.newaxis, :, :, np.newaxis])

            cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # _, jpeg = cv2.imencode(".jpg", frame)

        # return jpeg.tobytes()
        return frame


if __name__ == "__main__":
    cam = Camera()
    while True:
        frame = cam.get_frame()
        cv2.imshow('frame', frame)  # display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
