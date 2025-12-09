#!/bin/python3

from deepface import DeepFace   # Best model: Facenet512
import os
import cv2

def downscale(img_path : str, index : int):
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        max_dim = 1024
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        reduced = cv2.resize(img, (int(w * scale), int(h * scale)))
        format = img_path[img_path.find('.'):]
        cv2.imwrite(f'data/batch/{index}{format}', reduced)
    else:
        raise IOError(f"File {img_path} doesn't exists")

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
        print(f"Verifing '{file}'...")
        try:
            print(f"Found {len(DeepFace.extract_faces(file))} faces...")
            output = DeepFace.verify(subject, file, model_name='DeepFace')
            print(output)
            if output['verified']:
                print("Match!")
                matches.append(file)
            else:
                print("No match")
            print('')
        except IOError:
            exit(-1)
        except Exception:
            print("No face detected")

print(f"Subject '{subject}' found in {len(matches)} images:\n{matches}")