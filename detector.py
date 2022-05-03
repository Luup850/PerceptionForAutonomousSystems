#%%
import copy
import cv2
from matplotlib import pyplot as plt
import numpy as np

def sus_region_finder(sus_img, blank_sus_img):
    """Input: sus_img, blank_sus_img : numpy array
    Output: sus_score
    """

    org_sus_img = copy.deepcopy(sus_img)
    diff = copy.deepcopy(sus_img)
    blur = sus_blur(sus_img, 200)
    blank_blur = sus_blur(blank_sus_img, 200)
    cv2.absdiff(blur, blank_blur, diff)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _,thresh1 = cv2.threshold(diff_gray, 10, 255, cv2.THRESH_BINARY)
    #plt.imshow(diff_gray, cmap='gray')
    #plt.imshow(thresh1)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(diff_gray,kernel,iterations = 17)
    dilation = cv2.dilate(erosion,kernel,iterations = 17)
    _,thresh2 = cv2.threshold(dilation, 10, 255, cv2.THRESH_BINARY)
    #plt.imshow(thresh1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh2)

    width = np.shape(org_sus_img)[1]
    height = np.shape(org_sus_img)[0]

    output = []
    for i in range(num_labels):
        margin = 10
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

blank_image = cv2.imread("training_images/1585434750_438314676_Left.png")
non_blank = cv2.imread("training_images/purple_cup.png")
#non_blank = cv2.imread("training_images/black_box.png")
#print(np.shape(blank_image))
#(720,1280)
sus_region_finder(non_blank, blank_image)
cv2.waitKey(0)
# %%
