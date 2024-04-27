import cv2
import numpy as np
import os

previous_x = []
new_previous_x = []

filtered_frames = open("/Users/hirvapatel/Documents/Semester 6-2/Project /final/Dropwise-Condensation-Project/BG_Subtraction/filtered_frames.txt", 'w')

def identify_bounding_box(frame_num, x):
    new_file = open("coordinates_bounding_box/Frame_{}.txt".format(frame_num), 'r')
    for line in new_file:
        x1 = float(line.split()[0])
        x2 = float(line.split()[2])
        y1 = float(line.split()[1])
        y2 = float(line.split()[3])
        if x>x1 and x<x2:
            frame = cv2.imread('Frames/Frame_{}.png'.format(frame_num))
            filtered_frames.write("{}, {}, {}, {}, {}\n".format(frame_num, x1, y1, x2, y2))
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.imwrite('Filtered_Frames/Frame_{}.png'.format(frame_num), frame)
            return

for frame_num in range(1, 1016):
    path = "BG_sub_frames/Frame_{}.png".format(frame_num)
    if frame_num%50==0:
        print(frame_num)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_threshold = 3000  # Adjust this value according to your requirements
    big_white_patches = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            if y > 500 and x>200 and x<1300 and w<100 and x not in range(800,803): 
                flag = True
                for px in previous_x:
                    if abs(px-x) < 25:
                        flag = False
                if flag :
                    big_white_patches.append(contour)
                    identify_bounding_box(frame_num-1, x+(w/2))
                    new_previous_x.append(x)
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, big_white_patches, -1, (0, 255, 0), 2)
    cv2.imwrite('BG_sub_highlighted/Frame_{}.png'.format(frame_num), output_image)
    previous_x = new_previous_x
    new_previous_x = []
