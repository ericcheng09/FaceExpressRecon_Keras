from keras.models import load_model
import numpy as np


class FaceExpressionReconModel:

    Emotion = ["Angry", "Disgust", "Fear",
                 "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, path="FaceExpressRecon.h5"):
        try:
            self.model = load_model(path)
        except Exception as e:
            raise e

    def predictEmotion(self, image):
        self.pred = self.model.predict(image)
        return FaceExpressionReconModel.Emotion[np.argmax(self.pred)]
