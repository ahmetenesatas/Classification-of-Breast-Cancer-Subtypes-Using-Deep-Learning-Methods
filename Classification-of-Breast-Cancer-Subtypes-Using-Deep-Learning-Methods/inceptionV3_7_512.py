import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('new_dataset_7_512/train',
                                                 target_size = (512, 512),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('new_dataset_7_512/test',
                                            target_size = (512, 512),
                                            batch_size = 32,
                                            class_mode = 'categorical')

inceptionV3 = tf.keras.models.Sequential()

model = tf.keras.applications.InceptionV3(
                        include_top=False,
                        weights='imagenet',
                        input_shape=(512,512,3),
                        pooling=None,
                        classes=7,
                        classifier_activation="softmax")
model.trainable = False

inceptionV3.add(model)

inceptionV3.add(tf.keras.layers.Flatten())
inceptionV3.add(tf.keras.layers.Dense(units=512, activation='relu'))
inceptionV3.add(tf.keras.layers.Dense(units=7, activation='softmax'))

print(inceptionV3.summary())

inceptionV3.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
inceptionV3.fit(x= training_set, validation_data=test_set, epochs=50)

inceptionV3.save("saved_models/inceptionV3_7_512.h5")