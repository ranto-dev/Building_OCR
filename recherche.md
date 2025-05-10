Très bien ! Voici le **code final complet**, mis à jour pour fonctionner avec une **image réelle** et un **vocabulaire étendu** (lettres majuscules, minuscules, chiffres, accents, ponctuation).

---

## ✅ OCR avec vocabulaire complet et image réelle – Code Python prêt à l'emploi

### 🔠 Vocabulaire utilisé :

Inclut :
`a-z`, `A-Z`, `0-9`, accents (`éèàçù`), et ponctuation (`!?.,:;-_()`)

---

### 📁 Fichier : `ocr_image_inference.py`

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2

# === PARAMÈTRES ===
img_height = 32
img_width = 128
vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789éèàçù!?.,:;-_() ")
vocab_size = len(vocab)

# === 1. CHARGER UNE IMAGE ===
def load_image(filepath, img_height=32, img_width=128):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # canal unique
    img = np.expand_dims(img, axis=0)   # batch size 1
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
    x = layers.Dense(vocab_size + 1, activation='softmax')(x)  # +1 pour CTC blank

    return models.Model(inputs=input_img, outputs=x)

# === 3. DÉCODAGE PRÉDICTION ===
def predict_text(model, image, vocab, img_width):
    y_pred = model.predict(image)
    decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=[img_width // 4])
    result = ''.join([vocab[i] if i < len(vocab) else '' for i in decoded[0][0].numpy()])
    return result

# === 4. SCRIPT PRINCIPAL ===
if __name__ == "__main__":
    image_path = "image_test.png"  # <-- change ce chemin vers ton image réelle

    # Charger image
    image = load_image(image_path)

    # Construire le modèle (non entraîné ici)
    model = build_ocr_model((img_height, img_width, 1), vocab_size)

    # ⚠️ Résultat aléatoire car modèle non entraîné !
    predicted_text = predict_text(model, image, vocab, img_width)

    # Affichage image + texte prédit
    print("Texte prédit :", predicted_text)
    plt.imshow(image[0].squeeze(), cmap='gray')
    plt.title(f"Texte prédit : {predicted_text}")
    plt.axis('off')
    plt.show()
```

---

### ⚠️ Important :

Ce modèle **doit être entraîné** sur un vrai dataset pour donner des résultats valables. Actuellement, il prédit **n’importe quoi** (car les poids sont aléatoires).

---

Souhaites-tu que je t’aide ensuite à :

- Entraîner ce modèle avec un vrai jeu de données (IAM, SynthText, etc.) ?
- Ou intégrer **Tesseract OCR** pour extraire du texte sans entraînement ?

---

Bien sûr ! Pour concevoir un modèle OCR performant en utilisant le dataset SynthText, voici un guide détaillé pour :

1. **Télécharger et préparer le dataset SynthText**.
2. **Construire un modèle OCR basé sur CNN + LSTM + CTC avec TensorFlow**.
3. **Entraîner le modèle sur le dataset SynthText**.

---

## 1. Télécharger et préparer le dataset SynthText

Le dataset SynthText est une vaste collection d'images contenant du texte synthétique, idéale pour l'entraînement de modèles OCR.

### Étapes :

1. **Téléchargement du dataset** :

   - Téléchargez le fichier `SynthText.zip` depuis [ce lien](https://github.com/ankush-me/SynthText/issues/114).
   - Décompressez-le dans un répertoire, par exemple `SynthText/`.

2. **Téléchargement des annotations** :

   - Téléchargez le fichier `label.txt` (ou `shuffle_labels.txt` pour un sous-ensemble) depuis [ce lien](https://github.com/ankush-me/SynthText/issues/114).
   - Placez-le dans le répertoire `SynthText/`.

3. **Conversion des annotations** :

   Utilisez le script `synthtext_converter.py` pour convertir les annotations en un format utilisable pour l'entraînement :

   ```bash
   python tools/data/textrecog/synthtext_converter.py SynthText/gt.mat SynthText/ SynthText/synthtext/SynthText_patch_horizontal --n_proc 8
   ```

   Cette commande générera des images découpées et les associera à leurs annotations correspondantes.

---

## 2. Construire le modèle OCR avec TensorFlow

Nous allons utiliser un modèle basé sur des couches CNN pour l'extraction de caractéristiques, suivi de couches LSTM pour la séquence, et une couche Dense avec une fonction de perte CTC pour la reconnaissance du texte.

### Code du modèle :

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_ocr_model(input_shape, vocab_size):
    input_img = layers.Input(shape=input_shape, name='image')

    # Couches CNN
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # RNN avec LSTM
    new_shape = (input_shape[1] // 4, (input_shape[0] // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Couche Dense avec activation softmax
    x = layers.Dense(vocab_size + 1, activation='softmax')(x)  # +1 pour le "blank" du CTC

    return models.Model(inputs=input_img, outputs=x)
```

---

## 3. Entraîner le modèle sur SynthText

### Préparation des données :

- **Images** : Les images découpées sont stockées dans `SynthText/synthtext/SynthText_patch_horizontal/`.
- **Annotations** : Le fichier `label.txt` contient les transcriptions correspondantes.

### Chargement des données :

Utilisez une bibliothèque comme `tf.data` pour charger les images et leurs annotations de manière efficace.

### Fonction de perte CTC :

```python
def ctc_loss_fn(y_true, y_pred, input_length, label_length):
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
```

### Entraînement :

```python
# Paramètres
img_height, img_width = 32, 128
vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789éèàçù!?.,:;-_() ")
vocab_size = len(vocab)
batch_size = 32
epochs = 10

# Construction du modèle
model = build_ocr_model((img_height, img_width, 1), vocab_size)

# Compilation du modèle
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=ctc_loss_fn)

# Entraînement (à adapter selon votre méthode de chargement des données)
model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
```

---

## 4. Évaluation et utilisation du modèle

Après l'entraînement, vous pouvez évaluer le modèle sur un jeu de données de test et l'utiliser pour prédire du texte à partir d'images.

### Prédiction :

```python
def predict_text(model, image, vocab, img_width):
    y_pred = model.predict(image)
    decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=[img_width // 4])
    result = ''.join([vocab[i] if i < len(vocab) else '' for i in decoded[0][0].numpy()])
    return result
```

### Affichage du texte prédit :

```python
image = load_image("image_test.png")
predicted_text = predict_text(model, image, vocab, img_width)
print("Texte prédit :", predicted_text)
```

---

## 5. Ressources supplémentaires

- **MMOCR** : Une boîte à outils OCR basée sur PyTorch, compatible avec SynthText. Documentation : [https://mmocr.readthedocs.io/en/v0.3.0/datasets/recog.html](https://mmocr.readthedocs.io/en/v0.3.0/datasets/recog.html)
- **EasyOCR** : Une bibliothèque OCR facile à utiliser, compatible avec TensorFlow. Tutoriel : [https://www.freecodecamp.org/news/how-to-fine-tune-easyocr-with-a-synthetic-dataset](https://www.freecodecamp.org/news/how-to-fine-tune-easyocr-with-a-synthetic-dataset)

---

Souhaitez-vous que je vous aide à adapter ce code pour votre propre environnement ou à intégrer des fonctionnalités spécifiques ?
