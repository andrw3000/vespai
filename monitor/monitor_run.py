import os
import sys
import cv2
import time
import torch
import datetime
import argparse
import smtplib
import imghdr
from email.message import EmailMessage
from vibe import BackgroundSubtractor


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--min-motion-area', type=int, default=1.3e-4,
                    help="Pixel area in which motion is discounted.")
parser.add_argument('-b', '--brake', type=float, default=5.0,
                    help="Seconds break between each frame detection.")
parser.add_argument('-c', '--conf', type=float, default=0.8,
                    help="Positive detection confidence threshold.")
parser.add_argument('-d', '--dilation-strength', type=int, default=1,
                    help="No. dilations of active pixels, thickens detection")
parser.add_argument('-e', '--enable-email', action='store_true',
                    help="Activate email alerts.")
parser.add_argument('-f', '--frame-delay', type=int, default=50,
                    help="Number of frames to wait until next email alert.")
parser.add_argument('-m', '--motion-detection', action='store_true',
                    help="Force constant YOLOv5 detections.")
parser.add_argument('-p', '--print', action='store_true',
                    help="Enable print-outs.")
parser.add_argument('-ra', '--recipients', nargs='+',
                    default=['vespalert@outlook.com'],
                    help="Recipient email addresses to be alerted.")
parser.add_argument('-rd', '--root', type=str, default=os.getcwd(),
                    help="Root directory for project.")
parser.add_argument('-s', '--save', action='store_true',
                    help="Enable saving of output detections.")
parser.add_argument('-sd', '--save-dir', type=str,
                    default='monitor/detections',
                    help="Local directory to save outputs relative to root.")
parser.add_argument('-v', '--video', type=str, default=None,
                    help="Path to video file.")
args = parser.parse_args()


def motion_alert(frame, vibe_model=None, area_tol=1.3e4, dilation_strength=2):
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


def email_text(num_ah, num_eh, frame_id, dt):
    """Hornet alert data to be emailed and saved.

    Args:
        num_ah: [int] number of Asian hornets in frame
        num_eh: [int] number of European hornets in frame
        frame_id: [int] frame index
        dt: datetime object evaluated at hornet detection time
    """
    body = 'Identification of {num:d} hornets detected in frame ' \
           '{f:d} at {hr:02d}:{min:02d} on {d}/{m}/{y}.\n\n' \
           '\t{nah:d} Vespa velutina\n\t{neh:d} Vespa crabro'.format(
               num=num_ah + num_eh, f=frame_id, hr=dt.hour, min=dt.minute,
               d=dt.day, m=dt.month, y=dt.year, nah=num_ah, neh=num_eh,
           )
    filename = 'email-{nah:d}AH-{neh:d}EH-frame-{f:d}-{hr:02d}{min:02d}-' \
               '{d}-{m}-{y}.jpeg'.format(
                    f=frame_id, hr=dt.hour, min=dt.minute, d=dt.day,
                    m=dt.month, y=dt.year, nah=num_ah, neh=num_eh,
            )

    return body, filename


if __name__ == '__main__':

    # Identify saving directories
    if args.save:
        result_dir = os.path.join(args.root, args.save_dir, 'results')
        frame_dir = os.path.join(args.root, args.save_dir, 'frames')
        label_dir = os.path.join(args.root, args.save_dir, 'labels')
        email_dir = os.path.join(args.root, args.save_dir, 'emails')
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(email_dir, exist_ok=True)

    # Activate camera or video sample
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera or video file.")

    # Load YOLOv5 model
    yolo_dir = os.path.join(args.root, 'models/yolov5')
    model_dir = os.path.join(args.root, 'models/yolov5-params/yolov5s-2021.pt')
    sys.path.insert(0, yolo_dir)
    os.chdir(yolo_dir)
    model = torch.hub.load(
        yolo_dir, 'custom', path=model_dir, source='local', _verbose=False,
    )
    model.conf = args.conf

    # Set up email server
    if args.enable_email:
        server = smtplib.SMTP('smtp.office365.com', 587)
        # server.set_debuglevel(1)
        server.ehlo()
        server.starttls()
        server.login('vespalert@outlook.com', 'kitchenqu33n')

    # Collect first frame
    ret0, frame0 = cap.read()
    frame_id = 0
    if not ret0:
        raise RuntimeError("Input source didn't return the first frame.")

    # If detecting motion, instantiate ViBE (VIsual Background Extractor)
    if args.motion_detection:
        vibe_model = BackgroundSubtractor()
        vibe_model.init_history(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY))
    else:
        vibe_model = None

    # Iterate hornet monitor over an infinite loop
    frame_id = 1
    last_email_frame = -args.frame_delay
    while True:

        # Capture frame
        ret, next_frame = cap.read()
        timestamp = datetime.datetime.now()
        if not ret:
            print("No frame recieved. Exiting ...")
            break

        # Motion detection
        if args.motion_detection:
            run_detection, vibe_model = motion_alert(
                frame=next_frame,
                vibe_model=vibe_model,
                area_tol=args.min_motion_area,
                dilation_strength=args.dilation_strength,
            )

        else:
            run_detection = True

        # YOLOv5 object detection
        if run_detection:
            img = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
            results = model(img)
            predictions = results.pred[0]  # 0-th element of 1-elt list
            ah_count, eh_count = 0, 0
            for p in predictions:
                if p[-1] == 1:
                    ah_count += 1
                if p[-1] == 0:
                    eh_count += 1

            if ah_count + eh_count > 0:
                print(f'Positive hornet detections in frame #{frame_id}')
                results.render()  # updates results.imgs with boxes and labels
                img = cv2.cvtColor(results.imgs[0], cv2.COLOR_RGB2BGR)
                if args.save:
                    fname = os.path.join(frame_dir, f'frame-{frame_id}.jpeg')
                    lname = os.path.join(label_dir, f'frame-{frame_id}.txt')
                    rname = os.path.join(result_dir, f'frame-{frame_id}.jpeg')
                    cv2.imwrite(rname, img)
                    cv2.imwrite(fname, next_frame)
                    with open(lname, 'a') as f:
                        results_str = results.pandas().xyxy[0].to_string(
                            header=f'frame-{frame_id}', index=False,
                        )
                        f.write(results_str)

                time_passed = frame_id - last_email_frame >= args.frame_delay + 4
                if args.enable_email and time_passed:

                    # Write email
                    msg = EmailMessage()
                    msg['Subject'] = 'VespAI detection'
                    msg['From'] = 'vespalert@outlook.com'
                    msg['To'] = ', '.join(args.recipients)
                    dt_object = datetime.datetime.now()
                    email_body, email_filename = email_text(
                        ah_count, eh_count, frame_id, dt_object,
                    )
                    msg.set_content(email_body)

                    # Attach image
                    new_w = 512
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (new_w, int(h * new_w / w)))
                    attachment = os.path.join(email_dir, email_filename)
                    cv2.imwrite(attachment, img)
                    with open(attachment, 'rb') as f:
                        img_data = f.read()
                        msg.add_attachment(img_data, maintype='image',
                                           subtype=imghdr.what(None, img_data))

                    # Send email
                    server.send_message(msg)
                    last_email_frame = frame_id

        frame_id += 1
        time.sleep(args.brake)
    server.close()
