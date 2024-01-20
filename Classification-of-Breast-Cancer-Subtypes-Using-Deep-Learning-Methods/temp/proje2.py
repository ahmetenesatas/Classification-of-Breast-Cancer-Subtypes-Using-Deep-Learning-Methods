import os
import zipfile
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras
import cv2

#import K-Means
from sklearn.cluster import KMeans
# important metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def get_reference_dict(clusters,data_label):
    reference_label = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(clusters))):
        index = np.where(clusters == i,1,0)
        num = np.bincount(data_label[index==1]).argmax()
        reference_label[i] = num
    return reference_label
# Mapping predictions to original labels
def get_labels(clusters,reference_labels):
    temp_labels = np.random.rand(len(clusters))
    for i in range(len(clusters)):
        temp_labels[i] = reference_labels[clusters[i]]
    return temp_labels



data = []
label = []
path = "C:/Users/ergun/Desktop/ergunDosyalar/DERS/bilgisayarprojesi/dataset/test/"

n = 0
pb = 0
udh = 0
fea = 0
adh = 0
dcis = 0
ic = 0
IMG_SIZE = 128
for file in os.listdir(path):
    if file[0]=='0':
        for imgfil in os.listdir(path+file):
            img=cv2.imread(path+file+'/'+imgfil)
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            img=img.astype('float32')
            n+=1
            label.append("N")
            data.append(img)
    elif file[0]=='1':
        for imgFil in os.listdir(path+file):
            img=cv2.imread(path+file+'/'+imgFil)
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            img=img.astype('float32')
            pb+=1
            label.append("PB")
            data.append(img)
    elif file[0]=='2':
        for imgfil in os.listdir(path+file):
            img=cv2.imread(path+file+'/'+imgfil)
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            img=img.astype('float32')
            udh+=1
            label.append("UDH")
            data.append(img)
    elif file[0]=='3':
        for imgFil in os.listdir(path+file):
            img=cv2.imread(path+file+'/'+imgFil)
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            img=img.astype('float32')
            fea+=1
            label.append("FEA")
            data.append(img)
    elif file[0]=='4':
        for imgFil in os.listdir(path+file):
            img=cv2.imread(path+file+'/'+imgFil)
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            img=img.astype('float32')
            adh+=1
            label.append("ADH")
            data.append(img)
    elif file[0]=='5':
        for imgfil in os.listdir(path+file):
            img=cv2.imread(path+file+'/'+imgfil)
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            img=img.astype('float32')
            dcis+=1
            label.append("DCIS")
            data.append(img)
    elif file[0]=='6':
        for imgFil in os.listdir(path+file):
            img=cv2.imread(path+file+'/'+imgFil)
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            img=img.astype('float32')
            ic+=1
            label.append("IC")
            data.append(img)

data = np.array(data)

'''
data = np.array(data,dtype=np.uint8)
plt.imshow(data[0])
plt.show()'''

data_label = []
for i in label:
    if i=="N": data_label.append(0)
    elif i=="PB" : data_label.append(1)
    elif i=="UDH" : data_label.append(2)
    elif i=="FEA" : data_label.append(3)
    elif i=="ADH" : data_label.append(4)
    elif i=="DCIS" : data_label.append(5)
    elif i=="IC" : data_label.append(6)
data_label = np.array(data_label)

#data = data/255.0
reshaped_data = data.reshape(len(data),-1)

kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
clusters = kmeans.fit_predict(reshaped_data)

reference_labels = get_reference_dict(clusters,data_label)
predicted_labels = get_labels(clusters,reference_labels)
print(accuracy_score(predicted_labels,data_label))

#cluster_centers = np.array(cluster_centers, dtype=np.uint8)
image_cluster = kmeans.cluster_centers_.astype(np.uint8)[kmeans.labels_]
image_cluster = image_cluster.reshape(reshaped_data.shape[0], reshaped_data.shape[1])

plt.imshow(image_cluster)
plt.show()

# kalan kısım test amaçlı
'''
sse = []
list_k = [2,16,64,128,256]
for k in list_k:
    km = KMeans(n_clusters=k, n_init='auto')
    clusters = km.fit_predict(reshaped_data)
    
    sse.append(km.inertia_)
    reference_labels = get_reference_dict(clusters,data_label)
    predicted_labels = get_labels(clusters,reference_labels)
    print(f"Accuracy for k = {k}: ", accuracy_score(predicted_labels,data_label))

# Plot sse against k
plt.figure(figsize=(6,6))
plt.plot(list_k, sse, '-o')
plt.xlabel(f'Number of clusters {k}')
plt.ylabel('Sum of squared distance')

plt.show()'''