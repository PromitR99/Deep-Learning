import cv2
import matplotlib.pyplot as plt

imgpath="D:\\CODES\\OpenCV3\\DataSets\\4.2.01.tiff"
img=cv2.imread(imgpath, 1)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(imgRGB)
plt.imshow(imgRGB)
plt.title('Image')
plt.xticks([])
plt.yticks([])
plt.show()

'''
Creating an array of size: vertical_pixel_size*horizontal_pixel_size*number_of_colour channels x 1.
Along the vertical axis (from top to bottom): red channels, then green channels, then blue channels. 
'''

final_img = imgRGB.reshape((img.shape[0]*img.shape[1]*img.shape[2]), 1)
print(final_img)