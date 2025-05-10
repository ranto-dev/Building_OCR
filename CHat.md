```py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2

# === PARAMÈTRES ===
img_height = 32
img_width = 128
vocab = list("abcdefghijklmnopqrstuvwxyz ")
vocab_size = len(vocab)

# === 1. CHARGER UNE IMAGE ===
def load_image(filepath, img_height=32, img_width=128):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # Ajout canal
    img = np.expand_dims(img, axis=0)   # Ajout batch
    return img

# === 2. MODÈLE OCR (CNN + RNN + CTC) ===
def build_ocr_model(input_shape, vocab_size):
    input_img = layers.Input(shape=input_shape, name='image')

    # CNN
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # RNN
    new_shape = (img_width // 4, (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dense(vocab_size + 1, activation='softmax')(x)  # +1 pour le blank

    return models.Model(inputs=input_img, outputs=x)

# === 3. PRÉDICTION & DÉCODAGE ===
def predict_text(model, image):
    y_pred = model.predict(image)
    decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=[img_width // 4])
    return ''.join([vocab[i] if i < len(vocab) else '' for i in decoded[0][0].numpy()])

# === 4. MAIN SCRIPT ===
if __name__ == "__main__":
    image_path = "image_test.png"  # <--- remplace par ton chemin

    # Charger image
    image = load_image(image_path)

    # Créer modèle et charger poids si disponibles (ici non entraîné !)
    model = build_ocr_model((img_height, img_width, 1), vocab_size)

    # ⚠️ Le modèle n'est pas entraîné ici : les résultats seront aléatoires
    predicted = predict_text(model, image)

    # Affichage
    print("Texte prédit :", predicted)
    plt.imshow(image[0].squeeze(), cmap='gray')
    plt.title(f"Texte prédit : {predicted}")
    plt.axis('off')
    plt.show()
```
