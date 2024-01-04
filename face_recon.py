import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

## you need to download haarcascades from GitHub repo to use Classifiers.
faceCascade = cv2.CascadeClassifier('haarrcascade\haarcascade_frontalface_default.xml')
#use proper path above

while True:
    _, frame = capture.read()
    greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#greayscale works faster than coloured.
    faces = faceCascade.detectMultiScale(
        greyscale,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for x, y, face_width, face_height in faces:
        #frame, which can follow the face
        #remove # to use
        cv2.rectangle(frame, (x, y), (x + face_width, y + face_height), (0, 0, 255), 5)

        #using blur
        blur = cv2.blur(frame[y:y + face_height, x:x + face_width], ksize=(50,50))
        frame[y:y + face_height, x:x + face_width] = blur


    #coloured frame
    cv2.imshow('frame', frame)
    #grey frame, remove # to use
    #cv2.imshow('gray', greyscale)

#press esc button to break
    key = cv2.waitKey(50)
    if key == 27:
        break

capture.release()
