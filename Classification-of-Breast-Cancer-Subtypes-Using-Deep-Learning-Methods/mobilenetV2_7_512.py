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

enetV2L = tf.keras.models.Sequential()

model = tf.keras.applications.MobileNetV2(
                        alpha=1.0,
                        include_top=False,
                        weights='imagenet',
                        input_shape=(512,512,3),
                        pooling='max',
                        classes=7,
                        classifier_activation="softmax")
model.trainable = False

enetV2L.add(model)

enetV2L.add(tf.keras.layers.Flatten())
enetV2L.add(tf.keras.layers.Dense(units=512, activation='relu'))
enetV2L.add(tf.keras.layers.Dense(units=7, activation='softmax'))

print(enetV2L.summary())

enetV2L.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
enetV2L.fit(x= training_set, validation_data=test_set, epochs=50)

enetV2L.save("saved_models/mobilenetV2_7_512.h5")