import sys
import cv2
import datetime
from vibe import BackgroundSubtractor

# Parameters
min_area = 13000  # Adjust to seperate hornets from wasps
dilation_strength = 1  # To unify dijoint segmentation contours
save_dir = '/monitor/examples/wasps/'
save_outputs = True

# Instantiate ViBe (VIsual Background Extractor)
back_sub = BackgroundSubtractor()
ex = '/Users/Holmes/Research/Projects/vespai/datasets/' + \
     'videos/wasps.mp4'
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(ex)

if not cap.isOpened():
    print("Cannot open camera.")
    sys.exit()

frame_id = 0
while True:

    # Capture frame
    ret, frame = cap.read()
    timestamp = datetime.datetime.now()
    if not ret:
        print("No frame recieved. Exiting ...")
        break

    # Motion detection on frame
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame_id % 25 == 0:
        print(f'Frame number: {frame_id}')

    if frame_id == 0:
        back_sub.init_history(grey)

    # ViBe segmentation
    segmentation = back_sub.segmentation(grey)
    back_sub.update(grey, segmentation)
    segmentation = cv2.medianBlur(segmentation, 3)

    # Identify contours
    dilation = cv2.dilate(segmentation, None, iterations=dilation_strength)
    contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    # Draw contours to frame
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 4)

    # Loop over the contours
    for c in contours:

        # If the contour is too small (waspy), ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # Compute bounding box for contour, draw it on the frame,
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(grey, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if frame_id % 25 == 0 and save_outputs:
        cv2.imwrite(save_dir + f'contours-{frame_id}.jpeg', frame)
        cv2.imwrite(save_dir + f'detections-{frame_id}.jpeg', grey)
        cv2.imwrite(save_dir + f'segmentation-{frame_id}.jpeg', segmentation)
        cv2.imwrite(save_dir + f'dilation-{frame_id}.jpeg', dilation)

    cv2.imshow('Contours', frame)
    cv2.imshow('Detections', grey)
    cv2.imshow('Dilated segmentation', dilation)
    frame_id += 1

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()

# import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read()
# from detector.vibe import BackgroundSubtractor; bc = BackgroundSubtractor()
# cv2.imshow('Test', frame)
# ; cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
