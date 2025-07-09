import cv2
import imutils
from PIL import Image
import numpy as np
import PIL

# read an image as input using OpenCV
string= "C:/Users/padma/Downloads/"
li= ["coin","compass","coral","crystal","diamond","emerald","fossil","key","letter","shell","treasure_box"]

for a in li:
    image = Image.open(string+a+"-Photoroom.png").convert("RGBA")
    new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
    new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
    new_image.convert('RGB').save('test.png', "PNG")  # Save as JPEG
    image = cv2.imread("test.png")
    for i in range (360):
        Rotated_image = cv2.bitwise_not(image)
        Rotated_image = imutils.rotate(image, angle=i)
        Rotated_image = cv2.bitwise_not(image)
        img = cv2.cvtColor(Rotated_image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        im_pil = PIL.ImageOps.invert(im_pil)
        im_pil.save(string+"training_data/rotated/"+a+"/"+str(i)+".png")
        print(a,i)