import cv2
from run import *

video_path = '../Cu_DLC.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
ret, next_frame = cap.read()

k = 4 # Frames to be skipped

frame_num = 2
file = open('vectors_bounding_box.csv', 'w')
while ret:
    if frame_num%50==0:
      print(frame_num)
    cv2.imwrite('frame_1.png', frame)  
    cv2.imwrite('frame_2.png', next_frame)
    if (frame_num-1)%k==0:
        main('frame_2.png', 'frame_1.png', frame_num, file, k)
        frame = next_frame
    ret, nex_frame = cap.read()
    frame_num+=1