from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras

TRAINING_DIR = "dataset"
training_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size = (150, 150),
    class_mode = "categorical"
)
print("Traing data finished")

print(train_generator)

# Configuration des couches du réseau
model = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(150, 150, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(62)
])

# # Compilation du modele
model.compile(optimizer='adam',
            loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# # Entrainement du modele
model.fit_generator(train_generator, epochs=10, verbose=1)