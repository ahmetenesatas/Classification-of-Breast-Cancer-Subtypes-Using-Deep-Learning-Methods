import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

path = "C:/Users/ergun/Desktop/ergunDosyalar/DERS/bilgisayarprojesi/dataset"
kesit = 512
for ttv in os.listdir(path):
    for type in os.listdir(path+"/"+ttv):
        for file in os.listdir(path+"/"+ttv+"/"+type):
            image = cv2.imread(path+"/"+ttv+"/"+type+"/"+file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixel = image.reshape((-1, 3))

            K = 4
            kmeans = KMeans(n_clusters=K, n_init='auto')
            kmeans.fit(pixel)

            cluster_centers = kmeans.cluster_centers_

            segmented_image = np.zeros_like(pixel)
            for i in range(K):
                #print(cluster_centers[i])
                if(cluster_centers[i][0]<150 and cluster_centers[i][1]<70 and cluster_centers[i][2]<150):
                    segmented_image[kmeans.labels_ == i] = cluster_centers[i]

            segmented_image = segmented_image.reshape(image.shape)
            segmented_image = segmented_image.astype(np.uint8)

            '''
            plt.subplot(1,2,1)
            plt.imshow(data[i])
            plt.subplot(1,2,2)
            plt.imshow(segmented_image)
            plt.show()'''

            xfirst=0
            yfirst=0
            total=0
            for a in range(math.floor(image.shape[0]/kesit)-1):
                for b in range(math.floor(image.shape[1]/kesit)-1):
                    for x in range(kesit):
                        x_index = xfirst+x
                        for y in range(kesit):
                            y_index = yfirst+y
                            if segmented_image[y_index][x_index][0] != 0 or segmented_image[y_index][x_index][1]!=0 or segmented_image[y_index][x_index][2]!=0:
                                total +=1
                    if(total > kesit*kesit *4 /10): #above %40
                        #print(xfirst, yfirst)
                        cropped_image = image[yfirst:yfirst+kesit, xfirst:xfirst+kesit]
                        #plt.imshow(cropped_image)
                        #plt.show()
                        outputpath = "C:/Users/ergun/Desktop/ergunDosyalar/DERS/bilgisayarprojesi/new_dataset_7_"+str(kesit)+"/"+ttv+"/"+type+"/"+file[:-3]+str(xfirst)+str(yfirst)+'.png'
                        plt.imsave(outputpath, cropped_image)
                    total=0
                    xfirst+=kesit
                xfirst=0
                yfirst+=kesit