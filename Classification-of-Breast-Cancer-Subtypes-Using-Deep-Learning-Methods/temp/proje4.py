import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

image = cv2.imread('C:/Users/ergun/Desktop/ergunDosyalar/DERS/bilgisayarprojesi/dataset/train/0_N/BRACS_1857_N_2.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = cv2.resize(image,(1024,1024))
pixels = image.reshape((-1, 3))

K = 4

kmeans = KMeans(n_clusters=K, n_init='auto')
kmeans.fit(pixels)

cluster_centers = kmeans.cluster_centers_

segmented_image = np.zeros_like(pixels)
for i in range(K):
    print(cluster_centers[i])
    if(cluster_centers[i][0]<140 and cluster_centers[i][1]<70 and cluster_centers[i][2]<140):
        segmented_image[kmeans.labels_ == i] = cluster_centers[i]

segmented_image = segmented_image.reshape(image.shape)

segmented_image = segmented_image.astype(np.uint8)

plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(segmented_image)
plt.show()

xfirst=0
yfirst =0
total=0
kesit = 128
for a in range(math.floor(image.shape[0]/kesit)-1):
    for b in range(math.floor(image.shape[1]/kesit)-1):
        for x in range(kesit):
            x_index = xfirst+x
            for y in range(kesit):
                y_index = yfirst+y
                if segmented_image[y_index][x_index][0] != 0 or segmented_image[y_index][x_index][1]!=0 or segmented_image[y_index][x_index][2]!=0:
                    total +=1
        if(total > kesit*kesit/2):
            print(xfirst, yfirst)
            cropped_image = image[yfirst:yfirst+kesit, xfirst:xfirst+kesit]
            plt.imshow(cropped_image)
            plt.imsave('C:/Users/ergun/Desktop/'+str(xfirst)+str(yfirst)+'.png',cropped_image)
            plt.show()
        total=0
        xfirst+=kesit
    xfirst=0
    yfirst+=kesit