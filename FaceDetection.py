import cv2 
import keyboard

trained_face_data = cv2.CascadeClassifier('dummyface.xml')
# img = cv2.imread('photo1.jpg')
font = cv2.FONT_HERSHEY_SIMPLEX
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # getting coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, "Press Q to quit", (10,20), font , 1, (0,255,0),2)
    # display image
    cv2.imshow("Face Detection", frame)
    cv2.waitKey(1)

    if keyboard.is_pressed('q'):
        break
webcam.release()