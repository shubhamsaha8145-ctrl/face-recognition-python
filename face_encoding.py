import face_recognition

# load image
image = face_recognition.load_image_file("photo.jpg")

# get face encodings
encodings = face_recognition.face_encodings(image)

# check if face found
if len(encodings) > 0:
    print("Face encoding generated ✅")
    print("Total values:", len(encodings[0]))
    print(encodings[0])
else:
    print("No face found ❌")
