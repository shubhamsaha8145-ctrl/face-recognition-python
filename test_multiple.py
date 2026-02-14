import face_recognition
import cv2
import os

known_encodings = []
known_names = []

# Load known faces
for file in os.listdir("known_faces"):
    image = face_recognition.load_image_file("known_faces/" + file)
    encoding = face_recognition.face_encodings(image)[0]

    known_encodings.append(encoding)
    known_names.append(file.split(".")[0])

# Load test image
test_image = face_recognition.load_image_file("test2.jpg")
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)

    name = "Unknown"

    if True in matches:
        match_index = matches.index(True)
        name = known_names[match_index]

    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(test_image, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Result", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
