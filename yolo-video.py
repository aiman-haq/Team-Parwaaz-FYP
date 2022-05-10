from imutils.video import VideoStream
from imutils.video import FPS# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def average(lst):
    return round(sum(lst)/len(lst),4)

args = {'input': r"Videos\Vehicles\truck\truckdim2.mp4", 'output':r"output\truckdim2.avi", 'yolo': r"yolo-coco",'confidence': 0.5, 'threshold': 0.3}


labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1


dic  =  {'bicycle':[],'car': [],'motorcycle':[], 'train':[], 'airplane':[],'bus':[],'truck': [], 'boat':[]}
#dictree = {'tree':[]}
while True:
	
	(grabbed, frame) = vs.read()

	
	if not grabbed:
		break
	
	if W is None or H is None:
		(H, W) = frame.shape[:2]


	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	
	boxes = []
	confidences = []
	classIDs = []
	

	for output in layerOutputs:
		
		for detection in output:
			
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			
			if confidence > args["confidence"]:
				
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])
	
	if len(idxs) > 0:
		
		for i in idxs.flatten():
			
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			label = LABELS[classIDs[i]]

			if label in dic.keys():
				dic[label].append(confidences[i])
			print(confidences[i], LABELS[classIDs[i]])
	
			

	if writer is None:
		
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
		
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	writer.write(frame)
print(dic)
truck = dic['truck']
avg = average(truck)
print(avg)

#print(dictree)
#tree = dic['tree']
#avg = average(tree)
#print(avg)

print("[INFO] cleaning up...")
writer.release()
vs.release()
