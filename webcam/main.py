import cv2
import mediapipe as mp
from sklearn import svm
import pandas as pd
import os

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
df_test = pd.read_csv(dir_path + "\..\data\sign_mnist_test.csv")
df_test_x = df_test.loc[:, "pixel1":"pixel784"]
df_test_y = df_test.label

df_train = pd.read_csv(dir_path + "\..\data\sign_mnist_train.csv")
df_train_x = df_train.loc[:, "pixel1":"pixel784"]
df_train_y = df_train.label

svm = svm.SVC(kernel='linear')
svm.fit(df_train_x,df_train_y)
svm_predicted = svm.predict(df_test_x)

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            # print("HAY MANO")
            print(svm.predict(cv2.cvtColor(cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).flatten().reshape(1, -1)))
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        # else:
            # print("NO HAY MANO")

        return image

cap = cv2.VideoCapture(0)
tracker = handTracker()

while True:
    success,image = cap.read()
    image = tracker.handsFinder(image)
    # lmList = tracker.positionFinder(image)
    # if len(lmList) != 0:
    #     print(lmList[4])

    cv2.imshow("Video",image)
    cv2.waitKey(1)