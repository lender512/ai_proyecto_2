import cv2
import mediapipe as mp
from sklearn import svm
import numpy as np
import pandas as pd
import pywt
import os

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def reduce_dimension(letter, cuts, wavelet):
  for i in range(cuts):
    (letter, cD) = pywt.dwt(letter, wavelet)
  return letter

def vectorizar(matrix):
  return matrix.flatten()

def proccess_letters(dataset, wavelet, cuts = 4):
  
  data_X = []
  data_Y = []

  for letter_features in dataset:
      
      letter = letter_features[0]
      data_Y.append(letter)

      letter_features = reduce_dimension(letter_features[1:], cuts, wavelet)
      letter_features = vectorizar(letter_features)
      data_X.append(letter_features)

  return data_X, data_Y

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
df_test = pd.read_csv(dir_path + "\..\data\sign_mnist_test.csv")
df_test_x = df_test.loc[:, "pixel1":"pixel784"]
df_test_y = df_test.label

df_train = pd.read_csv(dir_path + "\..\data\sign_mnist_train.csv")
df_train_x = df_train.loc[:, "pixel1":"pixel784"]
df_train_y = df_train.label

svm = svm.SVC(kernel='rbf')
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

        # else:
            # print("NO HAY MANO")
            for hand_landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, hand_landmark, self.mpHands.HAND_CONNECTIONS)
                x = [landmark.x for landmark in hand_landmark.landmark]
                y = [landmark.y for landmark in hand_landmark.landmark]
                height, width, channels = image.shape
                center = np.array([np.mean(x)*width, np.mean(y)*height]).astype('int32')
                # center = center[0], center[1]+30
                cv2.circle(image, tuple(center), 10, (255,0,0), 1)  #for checking the center 

                a = int(max([int(max(x)*width-min(x)*width), int(max(y)*height-min(y)*height)])/2 + 40)
                pointA = (center[0]-a,center[1]-a)
                pointB = (center[0]+a,center[1]+a)
                cv2.rectangle(image, pointA, pointB, (255,0,0), 1)

                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (5,height-5)
                fontScale              = 1
                fontColor              = (255,255,255)
                thickness              = 5
                lineType               = 2

                

                cropped = image[pointA[0]:pointB[0], pointA[1]:pointB[1]]
                if cropped.size == 0: continue
                char = 65 + svm.predict(cv2.cvtColor(cv2.resize(cropped, (28, 28), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).flatten().reshape(1, -1))[0]
                cv2.putText(image,chr(char), 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)




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