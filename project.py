import cv2
from imutils import contours
import imutils
import numpy as np
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as dtc
from PIL import Image, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageOps
from PIL import Image, ImageOps
import datetime as Ctime
import xml.etree.ElementTree as xml

# Importing Dataset
data = pd.read_csv(".//storage//new_train.csv").values
clf = dtc()

# Training of Data
px_train = data[0:232, 1:]
train_label = data[0:232, 0]
clf.fit(px_train, train_label)

# Function to Predict the Image
def predict(file):
    img = Image.open(file)

    #Adding 2px border to the image
    img = ImageOps.expand(img, border=2, fill='black')

    #Resize the image to 28*28px
    new_width = 28
    new_height = 28
    img = img.resize((new_width, new_height), Image.ANTIALIAS)

    #Converting the image to Grey Scale
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = img.convert('L')
    img = ImageOps.invert(img)

    img.save('.//storage//28Pixel.png')

    #1D Pixel Array
    data = list(img.getdata())

    #Pixel Array to Decision Tree Classifier
    digit=clf.predict([data])

    return digit

try:
    #Reading the Image
    file=".//frames//"+str(input("Enter File Name :"))
    img = cv2.imread(file)

    #Resize the Image and applying the Blur Filter
    final=imutils.resize(img, height=800)
    blur = cv2.pyrMeanShiftFiltering(img, 29,57)

    #Converting BGR to HSV
    resize = imutils.resize(blur, height=800)
    hsv=cv2.cvtColor(resize,cv2.COLOR_BGR2HSV)

    #Range of Green Light
    greenLow=np.array([45,100,50],np.uint8)
    greenHigh=np.array([75,255,255],np.uint8)

    # Threshold the HSV image to get only Green colors
    green=cv2.inRange(hsv,greenLow,greenHigh)

    #Applying Gaussian Blur to the image
    kernel = np.ones((5, 5), np.uint8)
    green = cv2.GaussianBlur(green, (5, 5), 0)
    green = cv2.erode(green, kernel, iterations=1)

    #Dilate the image
    kernal=np.ones((3,3),'uint8')
    green=cv2.dilate(green,kernal)

    #Finding digits borders using Contours
    cnts = cv2.findContours(green.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    digitCnts = []

    # loop over the digit area candidates
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        #print(x,y,w,h)
        if w >= 5 and (h >= 20 and h <= 50):
            digitCnts.append(c)

    #Sorting Contours
    digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]
    digits = []

    #Iterating Over Contours and predicting the digits
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi =final[y:y + h, x:x + w]
        cv2.imwrite(".//storage//contoure.png", roi)
        digit=predict(".//storage//contoure.png")
        digits.append(digit)
        digit=0
    error=False
except Exception as e:
    print("!!! File Not Present !!!")
    error=True

if error==False :
    value=''.join(digits[0])+''.join(digits[1])
    print(value)
else :
    value="Null"

readings=list()

#Getting Current Date and Time
readings.append(str(Ctime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
readings.append(value)

#Writing Readings to the XML file
filename = './/storage//AC_Temp.xml'
root = xml.Element("Readings")

Time = xml.SubElement(root, "Time")
Time.text = readings[0]
Temp = xml.SubElement(root, "Temp")
Temp.text = readings[1]

tree = xml.ElementTree(root)
with open(filename, "wb") as fh:
    tree.write((fh))