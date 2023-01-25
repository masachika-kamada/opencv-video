import cv2
import numpy as np
import random2
import likelihood as li

cap = cv2.VideoCapture("movie/Tram.mp4")
ret, frame = cap.read()
h, w = frame.shape[:2]
np.random.seed(100)
Np = 500
px = np.zeros((Np), dtype=np.int64)
py = np.zeros((Np), dtype=np.int64)
lp = np.zeros((Np))
for i in range(Np):
    px[i] = int(np.random.uniform(0, w))
    py[i] = int(np.random.uniform(0, h))
obj = [0, 110, 160]
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    lp = li.likelihood(frame, px, py, obj, Np, sigma2=0.001)
    pxnew = np.array(random2.choices(population=px, weights=lp, k=Np)) + np.random.randint(-15, 15, Np)
    pynew = np.array(random2.choices(population=py, weights=lp, k=Np)) + np.random.randint(-15, 15, Np)
    px = np.where(pxnew > w-1, w-1, pxnew)
    py = np.where(pynew > h-1, h-1, pynew)
    px = np.where(px < 0, 0, px)
    py = np.where(py < 0, 0, py)
    for i in range(Np):
        cv2.circle(frame, (px[i], py[i]), 1, (0, 255, 0), 1)
    cv2.imshow("img", frame)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()
