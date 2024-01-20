import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('new_dataset_3_128/train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('new_dataset_3_128/test',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

resnet152 = tf.keras.models.Sequential()

model = tf.keras.applications.ResNet152V2(
                        include_top=False,
                        weights='imagenet',
                        input_shape=(128,128,3),
                        pooling='max',
                        classes=3,
                        classifier_activation="softmax")
model.trainable = False

resnet152.add(model)

resnet152.add(tf.keras.layers.Flatten())
resnet152.add(tf.keras.layers.Dense(units=128, activation='relu'))
resnet152.add(tf.keras.layers.Dense(units=3, activation='softmax'))

print(resnet152.summary())

resnet152.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
resnet152.fit(x= training_set, validation_data=test_set, epochs=50)

resnet152.save("saved_models/resnet152V2_3_128.h5")