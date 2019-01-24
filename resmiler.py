import cv2
import sys
import numpy as np

DEBUG = True if len(sys.argv) >= 2 and sys.argv[1] == 'd' else False

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
video_capture = cv2.VideoCapture(0)

smile_image = cv2.imread('smile.png', cv2.IMREAD_UNCHANGED)
sm_height, sm_width, sm_depth = smile_image.shape

while True:
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in Face:
        if DEBUG:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        smiles = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2,
                                              minNeighbors=22,
                                              minSize=(25, 25))
        for (ex, ey, ew, eh) in smiles:
            smile_roi = roi_color[ey:ey+eh, ex:ex+ew]
            if DEBUG:
                cv2.rectangle(roi_color, (ex-1, ey-1),
                              (ex+ew+2, ey+eh+2), (0, 255, 0), 2)
            scaled_smile_img = cv2.resize(smile_image, (ew, eh))
            # Do the alpha blending
            alpha = scaled_smile_img[:, :, 3] / 255
            alpha_mask = alpha[:, :, np.newaxis]
            alpha_mask = np.concatenate(
                (alpha_mask, alpha_mask, alpha_mask), axis=2)
            rgb = scaled_smile_img[:, :, :3]
            # cv2.imshow('mask', alpha_mask * 255)
            bg = (1 - alpha_mask) * smile_roi
            fg = alpha_mask * rgb
            roi_color[ey:ey+eh, ex:ex+ew] = cv2.add(bg, fg)
    # Display the resulting frame
    cv2.imshow('Face and Eye Detected', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cv2.waitKey(0)
video_capture.release()
cv2.destroyAllWindows()
