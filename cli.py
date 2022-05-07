from detector import SusDetector as Detector
from detector import makevideo
import cv2

# 1. Detector needs a blank image. These are found in 'training_images'
blank_image = cv2.imread("training_images/OclusionRight0.png")

white_cup = cv2.imread("training_images/white_cup.png")
book = cv2.imread("training_images/book.png")
purple_cup = cv2.imread("training_images/purple_cup.png")
black_box = cv2.imread("training_images/black_box.png")

# 2. Make detector and feed it the blank image
detector = Detector(blank_image)

# 3. Load the model. All models are saved in 'models'.
detector.load_model("models/sus_model_good.h5")

# 4. If you want to exclude areas of the image, you can do so by adding them to the 'exclude' list.
#    The way its done is by appending a tuple with the following format: (x, y, width, height)
#    This creates a square that the detector will ignore. Use paint to get pixel locations
detector.occlusion_area.append((658, 1074, 4, 687))
detector.occlusion_area.append((6, 347, 598, 719))
detector.occlusion_area.append((1233, 1279, 4, 687))
detector.occlusion_area.append((1059, 1280, 374, 720))

# 5. If you want to make a video, use the following function. It takes a folder full of images + a detector that is preconfigured.
makevideo("D:\\DataFolder\\Stereo_conveyor_with_occlusions\\Undistort_Right\\*", detector)
