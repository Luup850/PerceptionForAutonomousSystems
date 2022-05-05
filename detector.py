#%%
import copy
import cv2
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import glob
from sklearn.model_selection import train_test_split
import cv2

#Area (x start, x end, y start, y end)
def check_if_in_area(center : tuple, area : tuple):
    if(center[0] >= area[0] and center[0] <= area[1] and center[1] >= area[2] and center[1] <= area[3]):
        return True
    else:
        return False

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
        #cv2.rectangle(org_sus_img, (x_start, y_start), (x_end, y_end), (0,255,0), 2)
    #cv2.imshow("Sus", org_sus_img)
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

#print(np.shape(blank_image))
#(720,1280)
#sus_region_finder(non_blank, blank_image)
#cv2.waitKey(0)


# Actual detector
class SusDetector:

    def __init__(self, blank_image):
        self.blank_image = blank_image
        self.model = object
        self.labels = []
        self.IsCompiled = False

    def detect(self, img):
        if (self.IsCompiled == False):
            self.compile_model()
            self.IsCompiled = True

        sus_regions = sus_region_finder(img, self.blank_image)

        #copy_of_img = copy.deepcopy(img)
        #for region in sus_regions:
        #    cv2.rectangle(copy_of_img, (region[0], region[1]), (region[2], region[3]), (0,255,0), 2)
        #cv2.imshow("Sus", copy_of_img)

        regions = []

        # For sus debugging
        clone = copy.deepcopy(img)
        colors = [(255,0,0), (0,255,0), (0,0,255)]
        pos = []
        center = []

        occlusion_area = (725, 1032, 131, 679)

        for i,sus in enumerate(sus_regions):
            x_c = int((sus[2] + sus[0]) / 2)
            y_c = int((sus[3] + sus[1]) / 2)

            scale = int(max(sus[2] - sus[0], sus[3] - sus[1]) / 2)

            width = np.shape(img)[1]
            height = np.shape(img)[0]

            x_start = max(x_c - scale, 0)
            y_start = max(y_c - scale, 0)
            x_end = min(x_c + scale, width)
            y_end = min(y_c + scale, height)

            area = copy.deepcopy(img[y_start:y_end, x_start:x_end, :])
            area = cv2.resize(area, (32, 32))


            if(not check_if_in_area((x_c, y_c), occlusion_area)):
                regions.append(area)
                pos.append((x_start, y_start, x_end, y_end))
                center.append((x_c, y_c))

            
        regions = np.array(regions)
        res = self.model.predict(regions, batch_size=len(regions))

        for l,r in enumerate(res):
            largest = 0
            index = 0
            for i,j in enumerate(r):
                if(j > largest):
                    largest = j
                    index = i
            
            cv2.putText(clone, self.labels[index], (center[l][0], center[l][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[l % 3], 2)
            cv2.rectangle(clone, (pos[l][0], pos[l][1]), (pos[l][2], pos[l][3]), colors[l % 3], 2)

        ### Enable me to see the sus regions ###
        #cv2.imshow("Sus", clone)
 
        return (res, pos)

    
    def compile_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((7, 7)))
        #model.add(layers.Conv2D(4, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(3, activation='softmax'))

        self.model.summary()
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    def train(self, list_of_paths, labels, epoch):
        if (self.IsCompiled == False):
            self.compile_model()
            self.IsCompiled = True
        """
        Make sure the training images are scalable!
        """
        X = []
        y = []
        self.labels = copy.deepcopy(labels)
        for i,p in enumerate(list_of_paths):
            for filename in glob.glob(p):
                img = cv2.imread(filename)
                img = cv2.resize(img, (32, 32))
                #img.reshape(32, 32, 3)
                #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                X.append(img)
                y.append(i)
        
        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        self.model.fit(X_train, y_train, epochs=epoch, validation_data=(X_test, y_test))

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        if(self.IsCompiled == False):
            self.compile_model()
            self.IsCompiled = True
        self.model.load_weights(path)






#blank_image = cv2.imread("training_images/1585434750_438314676_Left.png")
#white_cup = cv2.imread("training_images/white_cup.png")
#book = cv2.imread("training_images/book.png")
#purple_cup = cv2.imread("training_images/purple_cup.png")
#black_box = cv2.imread("training_images/black_box.png")

#classes = ["Book", "Box", "Cup"]#, "Nothing"]
#paths = ["D:\\DataTraining\\book\\*", "D:\\DataTraining\\box\\*","D:\\DataTraining\\cup\\*"]#, "D:\\DataTraining\\nothing\\*"]
#classes = ["Cup"]
#paths = ["D:\\DataTraining\\cup\\*"]

#sus = SusDetector(blank_image)
#sus.labels = classes

#sus.train(paths, classes, 400)
#sus.save_model("C:\\Users\\Marcus\\Documents\\GitHub\\PerceptionForAutonomousSystems\\models\\sus_model.h5")

#sus.load_model("C:\\Users\\Marcus\\Documents\\GitHub\\PerceptionForAutonomousSystems\\models\\sus_model.h5")
#print("White Cup:")
#print(np.around(sus.detect(white_cup), decimals=2))
#cv2.waitKey(0)
#print("Book:")
#print(np.around(sus.detect(book), decimals=2))
#cv2.waitKey(0)
#print("Purple cup:")
#print(np.around(sus.detect(purple_cup), decimals=2))
#cv2.waitKey(0)
#print("Black box Cup:")
#print(np.around(sus.detect(black_box), decimals=2))
#cv2.waitKey(0)

#print(np.argmax(sus.detect(non_blank), axis = 1)[0])