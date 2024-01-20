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

K=16
kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto')

for imgData in data:
    imgData = np.array(imgData)
    imgData = imgData.reshape((-1, 3))
    clusters = kmeans.fit_predict(imgData)
    
    for i in range(K):
        array = np.zeros_like(imgData)
        if(kmeans.cluster_centers_[i]>100):
             array[kmeans.labels_ == i] = kmeans.cluster_centers_[i]
        
    array = array.reshape(IMG_SIZE,IMG_SIZE,3)
    array = np.array(array, dtype=np.uint8)
    plt.imshow(array)
    plt.show()