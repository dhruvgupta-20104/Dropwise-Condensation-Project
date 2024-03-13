import os
from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'Cu_DLC')
video_path_out = '{}_out_withoutlabel.mp4'.format(video_path)
video_path = video_path + ".mp4"

print(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5
frame_num = 0

while ret:
    frame_num+=1
    results = model(frame)[0]

    coordinates_file = open('..\coordinates_bounding_box\Frame_{}.txt'.format(frame_num), 'w')
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            coordinates_file.write("{} {} {} {}\n".format(int(x1), int(y1), int(x2), int(y2)))
            # cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        # cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()