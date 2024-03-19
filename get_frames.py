import cv2

video_path = 'Cu_DLC_out_withoutlabel.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

frame_num = 0

while ret:
    frame_num+=1
    if frame_num%50==0:
      print(frame_num)
    cv2.imwrite('Frames/Frame_{}.png'.format(frame_num), frame)
    ret, frame = cap.read()