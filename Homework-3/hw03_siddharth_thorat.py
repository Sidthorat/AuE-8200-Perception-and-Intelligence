# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:44:35 2022

@author: siddh
"""
###Importing the image libraries

from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
from PIL import Image
import cv2
from skimage import io
from skimage.color import rgb2gray


#1a....................................................................................................

img = Image.open('C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/MATLAB/Perception/HW_3_perception_thorat_siddharth/Lenna.jpg')
imgGray = img.convert('L')
imgGray.save('test_gray.jpg')

#Input image

input = cv2.imread('C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/MATLAB/Perception/HW_3_perception_thorat_siddharth/test_gray.jpg')
 
#Get input size
 
ho, wo = input.shape[:2]

#Desired "pixelated" size
 
w, h = (256, 256)
 
#Resize input to "pixelated" size
 
temp = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)

#Initialize output image
 
output = cv2.resize(temp, (wo, h), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("8bit_gray.jpg", output)



#1b....................................................................................................

image = Image.open('C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/MATLAB/Perception/HW_3_perception_thorat_siddharth/8bit_gray.jpg')
# WIDTH and HEIGHT are integers
r_image = image.resize((64, 64))
# r_image.save("resized_image.jpg")
plt.imshow(r_image)
plt.show()
plt.savefig('64_gray.jpg')


#1c....................................................................................................

 
#Transforming an image from color to grayscale
# Here we import the image file as an array of shape (nx, ny, nz)
image_file = 'C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/MATLAB/Perception/HW_3_perception_thorat_siddharth/Lenna.jpg'
input_image =  imread(image_file)  # this is the array representation of the input image
[nx, ny, nz] = np.shape(input_image)  # nx: height, ny: width, nz: colors (RGB)

# Extracting each one of the RGB components
r_img, g_img, b_img = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]

# The following operation will take weights and parameters to convert the color image to grayscale
gamma = 1.400  # a parameter
r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
g_s_i = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma

# This command will display the grayscale image alongside the original image
fig1 = plt.figure(1)
ax1, ax2 = fig1.add_subplot(121), fig1.add_subplot(122)
ax1.imshow(input_image)
ax2.imshow(g_s_i, cmap=plt.get_cmap('gray'))
fig1.show()

# Applying the Sobel operator
# The kernels Gx and Gy can be thought of as a differential operation in the "input_image" array in the directions x and y 
Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
# To know the shape of the input grayscale image
[rows, columns] = np.shape(g_s_i)  
 # initialization of the output image array (all elements are 0)
sobel_filtered_image = np.zeros(shape=(rows, columns)) 

# Now we "sweep" the image in both x and y directions and compute the output
for i in range(rows - 2):
    for j in range(columns - 2):
        gx = np.sum(np.multiply(Gx, g_s_i[i:i + 3, j:j + 3]))  # x direction
        gy = np.sum(np.multiply(Gy, g_s_i[i:i + 3, j:j + 3]))  # y direction
        sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"

# Display the original image and the Sobel filtered image
fig2 = plt.figure(2)
ax1, ax2 = fig2.add_subplot(121), fig2.add_subplot(122)
ax1.imshow(input_image)
ax2.imshow(sobel_filtered_image, cmap=plt.get_cmap('gray'))
fig2.show()
plt.title("Sobel kernal method")
#2a......................................................................................................................

#importing image librarires

# To read the input image
image = cv2.imread('C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/MATLAB/Perception/HW_3_perception_thorat_siddharth/test_gray.jpg')
color = ('b','g','r') 

# To find frequency of pixels in range (0-255)
for i,col in enumerate(color):
    histogram = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histogram,color = col)
    plt.xlim([0,256])

# To show the plotting graph of an image
plt.plot(histogram)
plt.title("lenna Gray Image Histogram Analysis")
plt.xlabel("Pixel values")
plt.ylabel("Frequency")

#2b......................................................................................................................

image = plt.imread('C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/MATLAB/Perception/HW_3_perception_thorat_siddharth/8bit_gray.jpg')

# To Flatten the image into 1-D: pixels
pixels = image.flatten()

# Generate a cumulative histogram
cdf, bins, patches = plt.hist(pixels, bins=256, range=(0, 256), normed=True, cumulative=True)
new_pixels = np.interp(pixels, bins[:-1], cdf*255)

# To Reshape new_pixels as a 2-D array: new_image
new_image = new_pixels.reshape(image.shape)

# To Display the new image with 'gray' color map
plt.subplot(2, 1, 1)
plt.imshow(new_image, cmap='gray')

# To Generate a histogram of the new pixels
plt.subplot(2, 1, 2)
pdf = plt.hist(new_pixels, bins=64, range=(0, 256), normed=False,
               color='red', alpha=0.4)
plt.grid('off')

# To Use plt.twinx() to overlay the CDF in the bottom subplot
_ = plt.twinx()
plt.xlim((0, 256))
plt.grid('off')

# Generate a cumulative histogram of the new pixels
cdf = plt.hist(new_pixels, bins=64, range=(0,256),
               cumulative=True, normed=True,
               color='orange', alpha=0.4)
plt.show()
plt.xlabel("Pixel values")
plt.ylabel("Frequency")


#2c......................................................................................................................

# Histogram Equalization is an image processing technique that adjusts the contrast of an image by using its histogram.

img = cv2.imread('C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/MATLAB/Perception/HW_3_perception_thorat_siddharth/test_gray.jpg')
plt.imshow(img, cmap='gray')
def histogram(image):
    
    h = image.shape[0]
    w = image.shape[1]
    
    histogram = np.zeros(256)
    
    for i in range(0, h):
        for j in range(0, w):
            pixel = image[i, j]
            histogram[pixel] = histogram[pixel] + 1
        
    for i in range(0, len(histogram)):
        histogram[i] = (histogram[i] / 480000)
    return histogram

def histogramEqualization(histogram, image):
    
    cdf = np.zeros(256)
    cdf[0] = histogram[0]
    
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + histogram[i]
        
    h = image.shape[0]
    w = image.shape[1]
        
    for i in range(0, h):
        for j in range(0, w):
            intensity = image[i, j]
            newIntensity = cdf[intensity]
            image[i, j] = int(newIntensity*255)
    
    return image

# Reading the Image
image_p = 'C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/MATLAB/Perception/HW_3_perception_thorat_siddharth/test_gray.jpg'

# Read the image and convert to grayscale
img = io.imread(image_p)
img = rgb2gray(img)

# To Convert the image to type uint8 and scale intensity values to the range 0-255.
im_unt = ((img - np.min(img)) * (1/(np.max(img) - np.min(img)) * 255)).astype('uint8')

print("Minimum intensity = " + str(np.min(im_unt)))
print("Maximum intensity = " + str(np.max(im_unt)))

# Showing the original image
figure1 = plt.figure(figsize = (10,10))
plt.gray()
plt.imshow(im_unt)
hist = histogram(im_unt)
plt.bar(np.arange(len(hist)), hist)
plt.show()
plt.title("Histogram Equilization")
plt.xlabel('Bins values')
plt.ylabel("Frequency")


#3a......................................................................................................................

import cv2
from matplotlib import pyplot as plt
img = cv2.imread('ParkingLot.jpg',0)
cv2.imshow("Parking Lot", img)
  
#histogram of an image
plt.hist(img.ravel(),256,[0,256])
plt.show()


# convert grayscale image to binary
(thresh, im_bw) = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#which determines the threshold automatically from the image using Otsu's method, or if you already know the threshold you can use:
im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

#Show binary image
cv2.imshow("Binary", im_bw)

cv2.waitKey(0)
cv2.destroyAllWindows()

#3(2)....................................................................................................................

img = cv2.imread("ParkingLot.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 75, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 3)

cv2.imshow("linesDetected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#3(4).....................................................................................................................


# Reading image
img2 = cv2.imread('ParkingLot.jpg', cv2.IMREAD_COLOR)
   
# Reading same image in another variable and 
# converting to gray scale.
img = cv2.imread('ParkingLot.jpg', cv2.IMREAD_GRAYSCALE)
   
# Converting image to a binary image 
# (black and white only image).
_,threshold = cv2.threshold(img, 110, 255, 
                            cv2.THRESH_BINARY)
   
# Detecting shapes in image by selecting region 
# with same colors or intensity.
contours,_=cv2.findContours(threshold, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
   
# Searching through every region selected to 
# find the required polygon.
for cnt in contours :
    area = cv2.contourArea(cnt)
   
    # Shortlisting the regions based on there area.
    if area > 400: 
        approx = cv2.approxPolyDP(cnt, 
                                  0.009 * cv2.arcLength(cnt, True), True)
   
        # Checking if the no. of sides of the selected region is 7.
        if(len(approx) == 7): 
            cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)
   
# Showing the image along with outlined arrow.
cv2.imshow('image2', img2) 
   
# Exiting the window if 'q' is pressed on the keyboard.
if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows()







