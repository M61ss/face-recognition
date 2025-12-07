#!/bin/python3

from deepface import DeepFace
import os

# Best model: Facenet512

subject : str = 'data/known/mattia.jpg'
try:
    print(f"Subject to be detected is '{subject}'")
    DeepFace.extract_faces(subject)
except:
    raise Exception(f"No face detected in '{subject}'")

files : list[str] = os.listdir('data/batch/')

matches = []
for file in files:
    file = 'data/batch/' + file
    if os.path.isfile(file):
        print(f"Verifing '{file}' ...")
        try:
            print(f"Found {len(DeepFace.extract_faces(file))} faces ...")
            output = DeepFace.verify(subject, file, model_name='DeepFace')
            print(output)
            if output['verified']:
                print("Match!")
                matches.append(file)
            else:
                print("No match")
            print()
        except Exception:
            print("No face detected")

print(f"Subject '{subject}' found in {len(matches)} images:\n{matches}")