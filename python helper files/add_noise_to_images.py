import cv2
import imutils
import numpy as np
import os
from PIL import Image

# read an image as input using OpenCV
string= "C:/Users/padma/Downloads/"
li= ["coin","compass","coral","crystal","diamond","emerald","fossil","key","letter","shell","treasure_box"]

def noisy(noise_typ,image):
    if noise_typ == "gaussian":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "saltpepper":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy




types=["gaussian","saltpepper","poisson","speckle"]
for typee in types:
    for a in li:
        for i in range (360):
            print(i, typee, a)
            imagereal= cv2.imread(string+"training_data/rotated/"+a+"/"+str(i)+".png")
            cv2.imwrite(string+"training_data/rotated_"+typee+"/"+a+"/"+str(i)+".png", noisy(typee, imagereal))
            