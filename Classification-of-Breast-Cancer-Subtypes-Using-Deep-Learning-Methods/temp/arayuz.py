import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
import tensorflow_addons as tfa

IMAGE_SIZE = 256
NUM_OF_CLASSES = 7
IMG_PATH = 'new_dataset_7_256/test/0_N/BRACS_1852_N_1.7681792.png'
label = '0_N'
MODEL = 'mobilenetV2'

classes_7 = ['0_N','1_PB','2_UDH','3_FEA','4_ADH','5_DCIS','6_IC']
classes_3 = ['0_B','1_A','2_M']

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory(f'new_dataset_{NUM_OF_CLASSES}_{IMAGE_SIZE}/train',
                                                 target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(f'new_dataset_{NUM_OF_CLASSES}_{IMAGE_SIZE}/test',
                                            target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model = tf.keras.models.load_model(f"saved_models/{MODEL}_{NUM_OF_CLASSES}_{IMAGE_SIZE}.h5")
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy', tfa.metrics.F1Score(num_classes=NUM_OF_CLASSES,average='macro',threshold=0.5), 
                                                                          tfa.metrics.CohenKappa(num_classes=NUM_OF_CLASSES)])

img = load_img(IMG_PATH, target_size=(IMAGE_SIZE,IMAGE_SIZE))
img_arr = img_to_array(img) / 255.0
pred = model.predict(np.expand_dims(img_arr, axis = 0))
if NUM_OF_CLASSES == 3:
    pred_lbl = classes_3[np.argmax(pred)]
elif NUM_OF_CLASSES == 7:
    pred_lbl = classes_7[np.argmax(pred)]

plt.title(f'Model: {MODEL} \nGercek Tip: {label} \nTahmin Tip: {pred_lbl}')
plt.imshow(np.array(img))
plt.show()

score = model.evaluate(x= training_set, verbose=0)

for i in range(4):
    print (f"{model.metrics_names[i]} : {score[i]}")