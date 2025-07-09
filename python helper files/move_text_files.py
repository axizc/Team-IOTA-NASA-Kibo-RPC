import os
import shutil

for i in range(40000):
    # os.rename("C:\\Users\\padma\\Downloads\\training_data\\mixed\\"+str(i)+".txt", "C:\\Users\\padma\\Downloads\\training_data\\mixedtxt\\"+str(i)+".txt")
    # os.replace("C:\\Users\\padma\\Downloads\\training_data\\mixed\\"+str(i)+".txt", "C:\\Users\\padma\\Downloads\\training_data\\mixedtxt\\"+str(i)+".txt")
    shutil.move("C:\\Users\\padma\\Downloads\\training_data\\mixed\\"+str(i+1)+".txt", "C:\\Users\\padma\\Downloads\\training_data\\mixedtxt\\"+str(i+1)+".txt")
