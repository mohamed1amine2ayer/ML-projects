import cv2
print("package imported") 
from IPython.core.display import Video
import cv2


cap=cv2.VideoCapture(1)
cap.set(10,100)
while True :
  succ,img =cap.read()
  cv2.imshow('ss',img)
  if cv2.waitKey(5) & 0xFF ==ord('q') :
    break
