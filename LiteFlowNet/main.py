import subprocess
import cv2
from run import *

video_path = '../Cu_DLC.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
ret, next_frame = cap.read()

frame_num = 0
file = open('vectors_bounding_box.csv', 'w')
while ret:
    frame_num+=1
    cv2.imwrite('frame_1.png', frame)
    cv2.imwrite('frame_2.png', next_frame)
    main('frame_1.png', 'frame_2.png', frame_num, file)
    frame = next_frame
    ret, nex_frame = cap.read()