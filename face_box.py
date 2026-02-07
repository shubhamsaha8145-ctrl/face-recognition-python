import face_recognition
import cv2

# open the photo
image = face_recognition.load_image_file("photo.jpg")

# find face
faces = face_recognition.face_locations(image)

# change color format
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# draw box on face
for top, right, bottom, left in faces:
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# resize image to fit screen
scale = 25  # try 50 or 60 if needed
width = int(image.shape[1] * scale / 100)
height = int(image.shape[0] * scale / 100)
image = cv2.resize(image, (width, height))

# show photo
cv2.imshow("Face", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
