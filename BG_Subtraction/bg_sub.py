import cv2

# Create a VideoCapture object to read the input video
cap = cv2.VideoCapture('../Cu_DLC.mp4')

# Create a background subtraction object
background_subtractor = cv2.createBackgroundSubtractorMOG2()
frame_num = 0

ret  = True

while ret:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction
    fg_mask = background_subtractor.apply(frame)

    if frame_num%10==0:
        print(frame_num)
    
    background_img = background_subtractor.getBackgroundImage()
    
    # Show the original frame and the foreground mask
    # cv2.imshow('Original', frame)
    # cv2.imshow('Foreground Mask', fg_mask)
    frame_num+=1
    cv2.imwrite('../BG_sub_frames/Frame_{}.png'.format(frame_num), fg_mask)
    # cv2.imshow('Foreground Mask', background_img)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
