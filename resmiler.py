import cv2
import sys
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
video_capture = cv2.VideoCapture(0)
while True:
  ret, img = video_capture.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  Face = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in Face:
     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
     roi_gray = gray[y:y+h, x:x+w]
     roi_color = img[y:y+h, x:x+w]
     smiles = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2,
                                                  minNeighbors=22,
                                                  minSize=(25, 25))
     for (ex,ey,ew,eh) in smiles:
       cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
  # Display the resulting frame
  cv2.imshow('Face and Eye Detected', img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
#cv2.waitKey(0)
video_capture.release()
cv2.destroyAllWindows()
