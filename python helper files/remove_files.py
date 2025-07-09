# Importing Required Modules
from rembg import remove
from PIL import Image
import cv2
import imutils
from PIL import Image
import numpy as np

# read an image as input using OpenCV
string= "C:/Users/padma/Downloads/"
li= ["coin","compass","coral","crystal","diamond","emerald","fossil","key","letter","shell","treasure_box"]

for a in li:
    for i in range (360):
        print(i, a)
        imagey= Image.open(string+"training_data/rotated/"+a+"/"+str(i)+".png")
        output = remove(imagey)
        output.save(string+"training_data/rotated_nobg/"+a+"/"+str(i)+".png")
        