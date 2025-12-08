from deepface import DeepFace
import cv2

img = cv2.imread("data/batch/20230820_193758.jpg")

# Resize
max_dim = 512
h, w = img.shape[:2]
scale = max_dim / max(h, w)
img_small = cv2.resize(img, (int(w*scale), int(h*scale)))

cv2.imwrite("data/batch/small.jpg", img_small)

print(DeepFace.verify(img1_path='data/known/mattia.jpg', 
                      img2_path='data/batch/small.jpg', 
                      model_name='VGG-Face', 
                      detector_backend='mtcnn'))