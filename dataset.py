import os
from PIL import Image, ImageOps
import cv2
import numpy as np
import csv
import pandas as pd

#Gettings all File Names from Directory digits
root1=".//storage//digits//"
all_files=os.listdir(root1)

#Directory to Store the Dataset images
root2=".//storage//pxdigi//"

Total=list()

#Permission to Store the DataSet Images
flag = int(input("Store DataSet Images [0/1] :"))

#Iterating over All Files
for file in all_files :
    #Image Imported
    img = Image.open(root1+file)

    #Converting Image to Grey Scale
    img = img.convert('L')

    #Applying 2px Border to the Image
    img=ImageOps.expand(img, border=2, fill='black')

    #Resize the image to 28*28 pixels
    new_width = 28
    new_height = 28
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img = ImageOps.invert(img)
    img = img.convert('L')
    img = ImageOps.invert(img)

    #Storing the DataSet Images
    if flag==1:
        img.save(root2+file)

    #Converting the Image 1D pixel Array
    data=np.asarray(img)
    data=data.reshape(-1)
    data = np.append(file[0],data)
    #print([data])
    Total.append(data)

#Creating a DataFrame to store all the 28*28 pixel values
my_df = pd.DataFrame(Total)

#Converting the DataFrame into CSV File
my_df.to_csv('.//storage//new_train.csv', index=False, header=False)

print("DataSet Created Successfully...!!!")
