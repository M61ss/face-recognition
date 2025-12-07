#!/bin/python3

from deepface import DeepFace
import os

# Best model: Facenet512

subject : str = 'data/known/mattia.jpg'
files : list[str] = os.listdir('data/')

outputs = []
for file in files:
    file_path = 'data/' + file
    if os.path.isfile(file_path):
        print(f"Verifing {file}")
        if DeepFace.verify(subject, file_path, model_name='Facenet512'):
            outputs.append(file_path)

print(f"Subject {subject} found in {len(outputs)} images:\n{outputs}")