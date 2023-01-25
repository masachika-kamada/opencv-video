import cv2

cap = cv2.VideoCapture("movie/Cruse.mp4")
ret, frame = cap.read()
h, w, ch = frame.shape

rct = (600, 500, 200, 200)
cv2.namedWindow("win", cv2.WINDOW_NORMAL)
cv2.resizeWindow("win", 1200, 800)
cri = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)
while True:
    threshold = 100
    ret, frame = cap.read()
    if ret == False:
        break
    img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_g, threshold, 255, cv2.THRESH_BINARY)
    ret, rct = cv2.CamShift(img_bin, rct, cri)
    x, y, w, h = rct
    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.imshow("win", frame)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()
