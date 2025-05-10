from PIL import Image
import pytesseract

# Ouvrir l'image
image_path = './data/img001-001.png'  # Remplace par le chemin de ton image
img = Image.open('./text-image.png')

# Utiliser Tesseract pour extraire le texte de l'image
text = pytesseract.image_to_string(img)

# Enregistrer le texte dans un fichier texte
print(text)