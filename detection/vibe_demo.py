import sys
import cv2
from vibe import BackgroundSubtractor

bs = BackgroundSubtractor()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    sys.exit()

frame_id = 0
while True:

    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("No frame recieved. Exiting ...")
        break

    # Motion detection on frame
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame_id % 100 == 0:
        print(f'Frame number: {frame_id}')

    if frame_id == 0:
        bs.init_history(grey)

    segmentation_map = bs.segmentation(grey)
    bs.update(grey, segmentation_map)
    segmentation_map = cv2.medianBlur(segmentation_map, 3)

    cv2.imshow('Colour Frame', frame)
    cv2.imshow('Greyscale', grey)
    cv2.imshow('Segmentation map', segmentation_map)
    frame_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()

# import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read()
# from detector.vibe import BackgroundSubtractor; bc = BackgroundSubtractor()
# cv2.imshow('Test', frame)
# ; cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
