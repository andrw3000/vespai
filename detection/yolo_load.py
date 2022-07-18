import os
import sys
import cv2
import glob
import torch


root_dir = '/Users/Holmes/Research/Projects/vespai'
yolo_dir = os.path.join(root_dir, 'models/yolov5')
model_dir = os.path.join(root_dir, 'models/yolov5-params/yolov5s-220715.pt')
save_dir = os.path.join(root_dir, 'models/yolov5-runs/test/yolov5s-220715')
os.makedirs(save_dir, exist_ok=True)
sys.path.insert(0, yolo_dir)
os.chdir(yolo_dir)

# Model
# Inference from various sources. For height=640, width=1280, ex RGB images:
#   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
#   URI:             = 'https://ultralytics.com/images/zidane.jpg'
#   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
#   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
#   numpy:           = np.zeros((640,1280,3))  # HWC
#   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
#   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
# model.load_state_dict(torch.load(weights_dir))

model = torch.hub.load(
    yolo_dir, 'custom', path=model_dir, source='local', _verbose=False,
)

# print(model)

# Images
files = glob.glob(
    os.path.join(root_dir, 'datasets/polygons-21/images/test/*.jpeg')
)
files = sorted(files)
imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files[:2]]

image_name = 'datasets/polygons-21/images/test/01FVYTPRMC4HQ3EB2E5C4K6X59.jpeg'
file_name = os.path.join(root_dir, image_name)
img = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)

# Inference
results = model(imgs)
a, b = results.pred

# Results
# results.print()
# results.save()  # or .show()

# print(results.xyxy[0])  # img1 predictions (tensor)
# fprint(results.pandas().xyxy[0])  # img1 predictions (pandas)

