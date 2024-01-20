import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('new_dataset_3_256/train',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('new_dataset_3_256/test',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'categorical')

enetV2L = tf.keras.models.Sequential()

model = tf.keras.applications.NASNetLarge(
                        include_top=False,
                        weights='imagenet',
                        input_shape=(256,256,3),
                        pooling='max',
                        classes=3,
                        classifier_activation="softmax")
model.trainable = False

enetV2L.add(model)

enetV2L.add(tf.keras.layers.Flatten())
enetV2L.add(tf.keras.layers.Dense(units=256, activation='relu'))
enetV2L.add(tf.keras.layers.Dense(units=3, activation='softmax'))

print(enetV2L.summary())

enetV2L.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
enetV2L.fit(x= training_set, validation_data=test_set, epochs=50)

enetV2L.save("saved_models/nasnetlarge_3_256.h5")