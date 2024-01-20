import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('new_dataset_7_128/train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('new_dataset_7_128/test',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

enetV2L = tf.keras.models.Sequential()

model = tf.keras.applications.NASNetLarge(
                        include_top=False,
                        weights='imagenet',
                        input_shape=(128,128,3),
                        pooling='max',
                        classes=7,
                        classifier_activation="softmax")
model.trainable = False

enetV2L.add(model)

enetV2L.add(tf.keras.layers.Flatten())
enetV2L.add(tf.keras.layers.Dense(units=128, activation='relu'))
enetV2L.add(tf.keras.layers.Dense(units=7, activation='softmax'))

print(enetV2L.summary())

enetV2L.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
enetV2L.fit(x= training_set, validation_data=test_set, epochs=50)

enetV2L.save("saved_models/nasnetlarge_7_128.h5")