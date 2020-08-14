import cv2

imgpath="D:\\CODES\\OpenCV3\\DataSets\\4.2.01.tiff"
img=cv2.imread(imgpath, 1)

final_img = img.reshape((img.shape[0]*img.shape[1]*img.shape[2]), 1)
print(final_img)