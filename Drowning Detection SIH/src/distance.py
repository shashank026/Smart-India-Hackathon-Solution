# install opencv "pip install opencv-python"
import cv2
import time
import os
from twilio.rest import Client


t = 0

# distance from camera to object(face) measured
# centimeter
Known_distance = 76.2

# width of face in the real world or Object Plane
# centimeter
Known_width = 14.3

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

# face detector object
face_detector = cv2.CascadeClassifier("/Users/vishxwas/Desktop/Distance_measurement_using_single_camera/haarcascade_frontalface_default.xml")

# focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):

    # finding the focal length
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

# distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):

    distance = (real_face_width * Focal_Length)/face_width_in_frame

    # return the distance
    return distance


def face_data(image):

    face_width = 0 # making face width to zero

    # converting color image to gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detecting face in the image
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    # looping through the faces detect in the image
    # getting coordinates x, y , width and height
    for (x, y, h, w) in faces:

        # draw the rectangle on the face
        cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)

        # getting face width in the pixels
        face_width = w

    # return the face width in pixel
    return face_width


# reading reference_image from directory
ref_image = cv2.imread("/Users/vishxwas/Desktop/Distance_measurement_using_single_camera/Ref_image.png")

# find the face width(pixels) in the reference_image
ref_image_face_width = face_data(ref_image)

# get the focal by calling "Focal_Length_Finder"
# face width in reference(pixels),
# Known_distance(centimeters),
# known_width(centimeters)
Focal_length_found = Focal_Length_Finder(
    Known_distance, Known_width, ref_image_face_width)

print(Focal_length_found)

# show the reference image
cv2.imshow("ref_image", ref_image)

# initialize the camera object so that we
# can get frame from it
# cap = cv2.VideoCapture("/Users/vishxwas/Downloads/Test2.mp4")
cap = cv2.VideoCapture(0)
# looping through frame, incoming from
# camera/video
while True:

    # reading the frame from camera
    _, frame = cap.read()

    # calling face_data function to find
    # the width of face(pixels) in the frame
    face_width_in_frame = face_data(frame)

    # check if the face is zero then not
    # find the distance
    if face_width_in_frame != 0:

        # finding the distance by calling function
        # Distance finder function need
        # these arguments the Focal_Length,
        # Known_width(centimeters),
        # and Known_distance(centimeters)
        Distance = Distance_finder(
            Focal_length_found, Known_width, face_width_in_frame)

        if Distance >= 70:
            cv2.putText(frame, 'High Risk',(30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
            t += 1
            converted_num = str(t)
            cv2.putText(frame, converted_num,(30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
            if t == 100:
                # account_sid = os.environ['AC61e99267368c2c3495078adfe47baf3f']
                # auth_token = os.environ['226669ce0ccbfec8871c4bcd2348eb3a']
                client = Client('AC61e99267368c2c3495078adfe47baf3f', '226669ce0ccbfec8871c4bcd2348eb3a')

                message = client.messages \
                    .create(
                    body="Person is drowning at coordinate: 28.7041° N, 77.1025° E 126",
                    from_='+19517449307',
                    to='+918881865469'
                )
                print(message.sid)
                t = 0
        else:
            t = 0
        #     count = time;
        # if count >= 1min -> sms send else -> not

        # draw line as background of text
        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)

        # Drawing Text on the screen
        cv2.putText(
            frame, f"Distance: {round(Distance,2)} CM", (30, 35),
            fonts, 0.6, GREEN, 2)

    # show the frame on the screen
    cv2.imshow("frame", frame)

    # quit the program if you press 'q' on keyboard
    if cv2.waitKey(1) == ord("q"):
        break

# closing the camera
cap.release()

# closing the windows that are opened
cv2.destroyAllWindows()