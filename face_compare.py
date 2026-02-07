"""
Face Recognition Comparison Script
Day 4–5 Project

This script:
1. Loads face images
2. Generates face encodings (128 values)
3. Compares same and different persons
4. Prints match result and distance
"""

import face_recognition

# ---------- LOAD IMAGES ----------
# Same person images
image_1a = face_recognition.load_image_file("person1_a.jpg")
image_1b = face_recognition.load_image_file("person1_b.jpg")

# Different person image
image_2 = face_recognition.load_image_file("person2.jpg")

# ---------- GENERATE FACE ENCODINGS ----------
encoding_1a = face_recognition.face_encodings(image_1a)[0]
encoding_1b = face_recognition.face_encodings(image_1b)[0]
encoding_2 = face_recognition.face_encodings(image_2)[0]

# ---------- SAME PERSON COMPARISON ----------
same_match = face_recognition.compare_faces([encoding_1a], encoding_1b)[0]
same_distance = face_recognition.face_distance([encoding_1a], encoding_1b)[0]

print("Same Person Comparison")
print("Match:", same_match)
print("Distance:", same_distance)

# ---------- DIFFERENT PERSON COMPARISON ----------
diff_match = face_recognition.compare_faces([encoding_1a], encoding_2)[0]
diff_distance = face_recognition.face_distance([encoding_1a], encoding_2)[0]

print("\nDifferent Person Comparison")
print("Match:", diff_match)
print("Distance:", diff_distance)
