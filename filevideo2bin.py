# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:28:44 2024

@author: Utilisateur
"""

from pathlib import Path

# Définir le chemin vers la vidéo
chemin_video = Path(r"C:\Users\Utilisateur\OneDrive\Documents\Ipsa 5\Télécom\test.mp4")  # Utilisation de raw string pour éviter l'erreur


# Vérifier si le fichier existe
if not chemin_video.exists():
    raise FileNotFoundError(f"Le fichier {chemin_video} est introuvable.")

# Lire le fichier vidéo en mode binaire
with open(chemin_video, "rb") as video_file:
    binary_data = video_file.read()

# Sauvegarder les données binaires dans un nouveau fichier
with open(r"C:\Users\Utilisateur\OneDrive\Documents\Ipsa 5\Télécom\video_binaire.bin", "wb") as binary_file:
    binary_file.write(binary_data)

print("La conversion en fichier binaire est terminée.")