import cv2
import dlib
import sys
import os
import face_recognition
# import cv2_imshow
from simple_facerec import SimpleFacerec
filename1 = r'C://Users/UZIMA//OneDrive/Desktop//FYP5//face-recognition/dataset//uzair-1.jpeg'
filename2 = r'C://Users/UZIMA//OneDrive/Desktop//FYP5//face-recognition/dataset//me2//uzair-6.jpeg'
img = cv2.imread(filename1)
# cv2.imshow("orignal image", img)
# cv2.waitKey(0)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread(filename2)
# cv2.imshow("test image",img2)
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

# # Encode faces from a folder
# sfr = SimpleFacerec()
# datasetpath = r'C://Users//UZIMA//OneDrive//Desktop//FYP5//face-recognition//dataset'
# sfr.load_encoding_images(datasetpath)

# # Load Camera
# cap = cv2.VideoCapture(0)


# while True:
#     ret, frame = cap.read()

#     # Detect Faces
#     face_locations, face_names = sfr.detect_known_faces(frame)
#     for face_loc, name in zip(face_locations, face_names):
#         y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

#         cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

#     cv2.imshow('frame', frame)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cap.release()
# # cv2.destroyAllWindows()