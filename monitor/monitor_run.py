import os
import sys
import cv2
import torch
import smtplib
import datetime
import argparse
from vibe import BackgroundSubtractor


def motion_alert(frame, vibe_model=None, area_tol=1.3e4, dilation_strength=1):
    """Returns positive bool if significant motion is detected.

    Args:
        frame: [numpy.ndarray] Next (BGR) frame to assess motion in
        vibe_model: [vibe object] ViBe background subtraction model
        area_tol: [int] Tolerance to reject small in-motion regions
        dilation_strength: [int] Number of dilations of motion pixels
    """

    if vibe_model:

        # ViBe segmentation
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        segmentation = vibe_model.segmentation(grey)
        vibe_model.update(grey, segmentation)
        segmentation = cv2.medianBlur(segmentation, 3)

        # Identify contours
        dilation = cv2.dilate(segmentation, None, iterations=dilation_strength)
        contours, hierarchy = cv2.findContours(
            dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Search contours for motion, ignoring areas smaller than area_tol
        motion_found = False
        for c in contours:
            if cv2.contourArea(c) > area_tol:
                motion_found = True
                break

        return motion_found, vibe_model

    # Declare motion_found=True if no vibe_bs
    else:
        return True, None


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--min-motion-area', type=int, default=1.3e-4,
                    help="Pixel area in which motion is discounted.")
parser.add_argument('-d', '--dilation-strength', type=int, default=1,
                    help="No. dilations of active pixels, thickens detection")
parser.add_argument('-m', '--motion-detection', action='store_true',
                    help="Force constant YOLOv5 detections.")
parser.add_argument('-p', '--print', action='store_true',
                    help="Enable print-outs.")
parser.add_argument('-r', '--root', type=str, default=os.getcwd(),
                    help="Root directory for project.")
parser.add_argument('-s', '--save', type=str, default='monitor/detections',
                    help="Local directory to save outputs relative to root.")
parser.add_argument('-ds', '--disable-save', action='store_true',
                    help="Disable saving of outputs.")
parser.add_argument('-v', '--video', type=str, default=None,
                    help="Path to video file.")
args = parser.parse_args()


if __name__ == '__main__':

    # Email server credentials
    gmail_user = 'vespalert@gmail.com'
    gmail_pw = 'kitchenqueen'

    # Identify save_dir
    if not args.disable_save:
        save_dir = os.path.join(args.root, args.save)
        os.makedirs(save_dir, exist_ok=True)

    # Activate camera or video sample
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera or video file.")

    # Load YOLOv5 model
    yolo_dir = os.path.join(args.root, 'models/yolov5')
    model_dir = os.path.join(args.root,
                             'models/yolov5-params/yolov5s-220715.pt')
    sys.path.insert(0, yolo_dir)
    os.chdir(yolo_dir)
    model = torch.hub.load(
        yolo_dir, 'custom', path=model_dir, source='local', _verbose=False,
    )

    # Earmark first frame
    ret0, frame0 = cap.read()
    frame_id = 0
    if not ret0:
        raise RuntimeError("Input source didn't return first frame")

    # If detecting motion, instantiate ViBe (VIsual Background Extractor)
    if args.motion_detection:
        motion_sensor = BackgroundSubtractor()
        motion_sensor.init_history(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY))
    else:
        motion_sensor = None

    # Iterate hornet monitor over an infinite loop
    while True:

        # Capture frame
        ret, next_frame = cap.read()
        timestamp = datetime.datetime.now()
        if not ret:
            print("No frame recieved. Exiting ...")
            break

        # Motion detection
        if args.motion_detection:
            run_detection, motion_sensor = motion_alert(
                frame=next_frame,
                vibe_model=motion_sensor,
                area_tol=args.min_motion_area,
                dilation_strength=args.dilation_strength,
            )

        else:
            run_detection = True

        # YOLOv5 object detection
        if run_detection:
            img = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
            results = model(img)
            num_positives_ah = 0
            for p in results.pred:
                if p[0, -1] == 1:
                    num_positives_ah += 1
            if num_positives_ah > 1:
                if not args.disable_save:
                    results.save(save_dir=save_dir)
                    h, w = img.shape[:2]
                img = cv2.resize(img, (h * 640 / w, 640))

                # Send email
                sent_from = gmail_user
                to = ['a.j.corbett@exeter.ac.uk', gmail_user]
                subject = 'VespAI AH detection'
                body = 'Vespa velitina detected in attached image.'
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.ehlo()
                server.login(gmail_user, gmail_pw)
                server.sendmail(sent_from, to, email_text)
                server.close()

        frame_id += 1
