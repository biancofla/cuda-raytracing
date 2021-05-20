from natsort import natsorted, ns
from cv2 import cv2
import os

images_folder = './'

images = [img for img in os.listdir(images_folder) if img.endswith(".bmp")]
images = natsorted(images, key=lambda y: y.lower())  
frame = cv2.imread(os.path.join(images_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter('animation.avi', 0, 25, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(images_folder, image)))

cv2.destroyAllWindows()
video.release()