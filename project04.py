import cv2
import numpy as np

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.resizeWindow("img", 1200, 800)
cap = cv2.VideoCapture("movie/People.mp4")
ret, frame = cap.read()
h, w, ch = frame.shape
frame_back = np.zeros((h, w, ch), dtype=np.float32)
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame_diff = cv2.absdiff(frame.astype(np.float32), frame_back)
    cv2.accumulateWeighted(frame, frame_back, 0.03)
    cv2.imshow("img", frame_diff.astype(np.uint8))
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()
