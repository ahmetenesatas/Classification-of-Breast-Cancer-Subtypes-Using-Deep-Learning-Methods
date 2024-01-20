import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

data = []
pixels = []
path = "C:/Users/ergun/Desktop/ergunDosyalar/DERS/bilgisayarprojesi/dataset/train/1_PB/"

IMG_SIZE = 1024
for file in os.listdir(path):
    image = cv2.imread(path+file)
    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel = image.reshape((-1, 3))
    data.append(image)
    pixels.append(pixel)
img_arr = np.array(data)
pixel_arr = np.array(pixels)


segmented_arr = []
for i in range(len(data)):
    K = 4
    kmeans = KMeans(n_clusters=K, n_init='auto')
    kmeans.fit(pixels[i])

    cluster_centers = kmeans.cluster_centers_

    segmented_image = np.zeros_like(pixel_arr[i])
    for j in range(K):
        #print(cluster_centers[j])
        if(cluster_centers[j][0]<150):
            segmented_image[kmeans.labels_ == j] = cluster_centers[j]

    segmented_image = segmented_image.reshape(img_arr[i].shape)
    segmented_image = segmented_image.astype(np.uint8)
    segmented_arr.append(segmented_image)
    '''
    plt.subplot(1,2,1)
    plt.imshow(data[i])
    plt.subplot(1,2,2)
    plt.imshow(segmented_image)
    plt.show()'''


for i in range(len(img_arr)):
    xfirst=0
    yfirst =0
    total=0
    for a in range(math.floor(img_arr[i].shape[0]/64)-1):
        for b in range(math.floor(img_arr[i].shape[1]/64)-1):
            for x in range(64):
                x_index = xfirst+x
                for y in range(64):
                    y_index = yfirst+y
                    if segmented_arr[i][y_index][x_index][0] != 0 or segmented_arr[i][y_index][x_index][1]!=0 or segmented_arr[i][y_index][x_index][2]!=0:
                        total +=1
            if(total > 2048): # above %50
                print(i, xfirst, yfirst)
                cropped_image = img_arr[i][yfirst:yfirst+64, xfirst:xfirst+64]
                plt.imshow(cropped_image)
                plt.imsave('C:/Users/ergun/Desktop/ergunDosyalar/DERS/bilgisayarprojesi/new_dataset/train/1_PB/'+str(i)+str(xfirst)+str(yfirst)+'.png',cropped_image)
                #plt.show()
            total=0
            xfirst+=64
        xfirst=0
        yfirst+=64