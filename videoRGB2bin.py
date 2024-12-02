import cv2
import os
import numpy as np



# Vérifier le contenu du dossier "data"
folder = r"C:\Users\Utilisateur\OneDrive\Documents\Ipsa 5\Télécom"

if not os.path.exists(folder):
    print(f"Erreur : Le dossier {folder} n'existe pas.")
else:
    files = os.listdir(folder)
    if not files:
        print(f"Erreur : Le dossier {folder} est vide.")
    else:
        print(f"Fichiers trouvés dans {folder} : {files}")



def video_to_binary(input_video, output_binary_file):
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo {input_video}")
        return

    frame_count = 0
    binary_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"Traitement de la frame {frame_count}...")

        # Convertir l'image en tableau 2D de valeurs binaires
        binary_frame = []
        for pixel in frame.reshape(-1, 3):  # Remise à plat du tableau (R, G, B)
            binary_pixel = ''.join([f'{channel:08b}' for channel in pixel])  # Convertit chaque canal (R, G, B) en binaire
            binary_frame.append(binary_pixel)

        # Ajouter cette frame à la liste globale
        binary_data.append(''.join(binary_frame))

    # Libérer la vidéo et fermer
    cap.release()

    # Sauvegarder le résultat en binaire dans un fichier texte
    with open(output_binary_file, 'w', encoding='utf-8') as f:
        for frame_binary in binary_data:
            f.write(frame_binary + '\n')

    print(f"Conversion terminée. Le fichier binaire a été enregistré sous {output_binary_file}.")



# Utilisation de la fonction
input_video_path = r"C:\Users\Utilisateur\OneDrive\Documents\Ipsa 5\Télécom\test.mp4" # Chemin de la vidéo en entrée
output_binary_path = "output_binary.txt"  # Chemin du fichier binaire en sortie
video_to_binary(input_video_path, output_binary_path)
