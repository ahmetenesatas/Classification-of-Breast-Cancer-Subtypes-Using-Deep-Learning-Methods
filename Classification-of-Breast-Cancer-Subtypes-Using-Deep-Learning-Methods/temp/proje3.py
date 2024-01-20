import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from PIL import Image

data = []
path = "C:/Users/ergun/Desktop/ergunDosyalar/DERS/bilgisayarprojesi/dataset/val/"

IMG_SIZE = 1024
kesit = 64
for file in os.listdir(path):
    for imgfil in os.listdir(path+file):
        img=cv2.imread(path+file+'/'+imgfil)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img=img.astype('uint8')
        data.append(img)

data = np.array(data)

kmeans = KMeans(n_clusters=4, random_state=0, n_init='auto')

for imgData in data:
    imgData = np.array(imgData)
    imgData = imgData.reshape((-1, 3))
    clusters = kmeans.fit_predict(imgData)
    
    for center in kmeans.cluster_centers_:
        #print(imgData[center])
        orgImage = imgData.reshape(IMG_SIZE,IMG_SIZE,3)

        center1= center[0]
        center2= center[1]
        print(f"Center1: {center1}, Center2: {center2}")

        cropped = Image.fromarray(orgImage.astype(np.uint8)).crop((center1+kesit,center2-kesit, IMG_SIZE-center1-kesit,IMG_SIZE-center2-kesit))
        plt.imshow(cropped)
        plt.show()