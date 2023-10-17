# This is a demo of running face recognition on a Raspberry Pi.
# This program will print out the names of anyone it recognizes to the console.

# To run this, you need a Raspberry Pi 2 (or greater) with face_recognition and
# the picamera[array] module installed.
# You can follow this installation instructions to get your RPi set up:
# https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65

import face_recognition
#import picamera
import cv2 as cv
import numpy as np

# IF Raspberrypi
# Get a reference to the Raspberry Pi camera.
# If this fails, make sure you have a camera connected to the RPi and that you
# enabled your camera in raspi-config and rebooted first.
#camera = picamera.PiCamera()
#camera.resolution = (320, 240)
#output = np.empty((240, 320, 3), dtype=np.uint8)
# ELSE webcam/opencv
#Get a reference from webcam
camera = cv.VideoCapture(0)
# end if

if not camera.isOpened():
    print("Cannot open camera")
    exit()



# Load a sample picture and learn how to recognize it.
print("Loading known face image(s)")
donatien_image = face_recognition.load_image_file("./images/donatien.jpg")
donatien_face_encoding = face_recognition.face_encodings(donatien_image)[0]
claire_image = face_recognition.load_image_file("./images/claire.jpg")
claire_face_encoding = face_recognition.face_encodings(claire_image)[0]
vianney_image = face_recognition.load_image_file("./images/vianney.jpg")
vianney_face_encoding = face_recognition.face_encodings(vianney_image)[0]
domitille_image = face_recognition.load_image_file("./images/domitille.jpg")
domitille_face_encoding = face_recognition.face_encodings(domitille_image)[0]
louise_image = face_recognition.load_image_file("./images/louise.jpg")
louise_face_encoding = face_recognition.face_encodings(louise_image)[0]
# Create arrays of known face encodings and their names
known_face_encodings = [
    donatien_face_encoding,
    claire_face_encoding,
    vianney_face_encoding,
    domitille_face_encoding,
    louise_face_encoding
]
known_face_names = [
    "Donatien",
    "Claire",
    "Vianney",
    "Domitille",
    "Louise"
]
know_face_authorisation = [
    1,
    1,
    1,
    0,
    0,
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

while True:
    print("Capturing image.")
    # IF Raspberrypi
    # Grab a single frame of video from the RPi camera as a numpy array
    #camera.capture(output, format="rgb")
    # ELSE webcam/opencv
    #  Grab a single frame from webcam
    ret, frame = camera.read()
    output = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #end if

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(output)
    face_encodings = face_recognition.face_encodings(output, face_locations)

    face_names = []
    face_authorisations = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        auth = 0

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            auth =  know_face_authorisation[best_match_index]

        face_names.append(name)
        face_authorisations.append(auth)

        # Display the results
    for (top, right, bottom, left), name, authorisation in zip(face_locations, face_names, face_authorisations):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face

        # Draw a label with a name below the face
        if authorisation :
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv.FILLED)
        else :
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv.imshow('frame', frame)

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
camera.release()
cv.destroyAllWindows()