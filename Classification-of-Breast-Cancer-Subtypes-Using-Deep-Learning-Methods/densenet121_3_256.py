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

densenet = tf.keras.models.Sequential()

model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(256,256,3),
        pooling='max',
        classes=3,
        classifier_activation="softmax")
model.trainable = False

densenet.add(model)
densenet.add(tf.keras.layers.Flatten())
densenet.add(tf.keras.layers.Dense(units=256, activation='relu'))
densenet.add(tf.keras.layers.Dense(units=3, activation='softmax'))

print(densenet.summary())

densenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
densenet.fit(x= training_set, validation_data=test_set, epochs=50)

densenet.save("saved_models/densenet121_3_256.h5")