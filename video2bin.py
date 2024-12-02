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







def video_to_binary(input_video, output_binary_file, threshold=128):
    """
    Convertit une vidéo .mp4 en binaire et écrit la sortie dans un fichier binaire.

    Parameters:
    - input_video (str): Chemin du fichier vidéo d'entrée.
    - output_binary_file (str): Chemin du fichier de sortie binaire.
    - threshold (int): Seuil de binarisation (0-255), par défaut 128.
    """

    # Ouvrir la vidéo avec OpenCV
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir le fichier vidéo.")
        return

    # Ouvrir le fichier binaire en mode écriture
    with open(output_binary_file, 'w', encoding='utf-8') as bin_file:
        frame_count = 0
        while True:
            # Lire une frame de la vidéo
            ret, frame = cap.read()

            if not ret:  # Si aucune frame n'est retournée, fin de la vidéo
                break

            # Convertir l'image en niveaux de gris
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Binariser l'image en fonction du seuil
            binary_frame = (gray_frame >= threshold).astype(np.uint8)

            # Convertir chaque pixel binaire (0 ou 1) en une chaîne binaire
            binary_string = ''.join(map(str, binary_frame.flatten()))

            # Écrire la chaîne binaire dans le fichier de sortie
            bin_file.write(binary_string)

            print(f"Frame {frame_count} convertie et écrite en binaire.")
            frame_count += 1

    # Libérer la capture vidéo et fermer le fichier
    cap.release()
    print(f"Conversion terminée. Le fichier binaire est enregistré sous : {output_binary_file}")


# Utilisation de la fonction
input_video = r"C:\Users\Utilisateur\OneDrive\Documents\Ipsa 5\Télécom\test.mp4" # Chemin de la vidéo en entrée
output_binary_file = "output.bin"  # Chemin du fichier binaire en sortie
video_to_binary(input_video, output_binary_file)
