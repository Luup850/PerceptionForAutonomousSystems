#%%
import copy
import cv2
from cv2 import HOGDESCRIPTOR_L2HYS
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import glob
from sklearn.model_selection import train_test_split
import cv2

def sus_region_finder(sus_img, blank_sus_img):
    """Input: sus_img, blank_sus_img : numpy array
    Output: sus_score
    """

    org_sus_img = copy.deepcopy(sus_img)
    diff = copy.deepcopy(sus_img)
    blur = sus_blur(sus_img, 100)
    blank_blur = sus_blur(blank_sus_img, 100)
    cv2.absdiff(blur, blank_blur, diff)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _,thresh1 = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)
    #plt.imshow(diff_gray, cmap='gray')
    #cv2.imshow("Sus", thresh1)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(diff_gray,kernel,iterations = 17)
    dilation = cv2.dilate(erosion,kernel,iterations = 19)
    _,thresh2 = cv2.threshold(dilation, 10, 255, cv2.THRESH_BINARY)
    #plt.imshow(thresh1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh2)

    width = np.shape(org_sus_img)[1]
    height = np.shape(org_sus_img)[0]

    output = []
    for i in range(num_labels):
        if (i == 0):
            continue
        margin = 0
        x_start = max(stats[i,0] - margin, 0)
        y_start = max(stats[i,1] - margin, 0)
        x_end = min(stats[i,0]+stats[i,2] + margin, width)
        y_end = min(stats[i,1]+stats[i,3] + margin, height)
        output.append((x_start, y_start, x_end, y_end))
        cv2.rectangle(org_sus_img, (x_start, y_start), (x_end, y_end), (0,255,0), 2)
    cv2.imshow("Sus", org_sus_img)
    return output
    

# Susifies and image by blurring it
def sus_blur(img, blur_amount):
    for i in range(blur_amount):
        img = cv2.GaussianBlur(img,(5,5),0)
    return img

def errode_dilate(img, kernel_size, ite):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = ite)
    dilation = cv2.dilate(erosion,kernel,iterations = ite)
    return dilation

blank_image = cv2.imread("training_images/1585434750_438314676_Left.png")
#non_blank = cv2.imread("training_images/white_cup.png")
#non_blank = cv2.imread("training_images/book.png")
#non_blank = cv2.imread("training_images/purple_cup.png")
non_blank = cv2.imread("training_images/black_box.png")
#print(np.shape(blank_image))
#(720,1280)
sus_region_finder(non_blank, blank_image)
cv2.waitKey(0)


# Actual detector
class SusDetector:

    def __init__(self, blank_image):
        self.blank_image = blank_image
        self.model = object

    def detect(self, img):
        sus_regions = sus_region_finder(img, self.blank_image)

        for sus in sus_regions:
            x_c = int((sus[2] - sus[0]) / 2)
            y_c = int((sus[3] - sus[1]) / 2)

            scale = int(max(sus[2] - sus[0], sus[3] - sus[1]) / 2)

            width = np.shape(img)[1]
            height = np.shape(img)[0]

            x_start = max(x_c - scale, 0)
            y_start = max(y_c - scale, 0)
            x_end = min(x_c + scale, width)
            y_end = min(y_c + scale, height)

            area = img[x_start:x_end, y_start:y_end]
            area = cv2.resize(area, (32, 32))
            
            return self.model.predict(area)

    
    def compile_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
        self.model.add(layers.MaxPooling2D((7, 7)))
        #model.add(layers.Conv2D(4, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(4, activation='softmax'))

        self.model.summary()
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    def train(self, train_images, train_labels):
        pass
        #self.model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))