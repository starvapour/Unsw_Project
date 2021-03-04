import imageio
import cv2
import numpy as np

label = np.array(imageio.mimread("read.gif"))[0]
cv2.imwrite("write.png",label)
