# USAGE
# python detect_faces_video.py --video VIDEOFILENAME.mp4 --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import os
from imutils.video import FileVideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
  help="path to input video file (optional)")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# get args and concatenate with current directory
# get current directory
dirname, filename = os.path.split(os.path.abspath(__file__))
prototxt = os.path.join(dirname, args["prototxt"])
model = os.path.join(dirname, args["model"])

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream and allow the cammera sensor to warmup
if args["video"] is None:
    print("[INFO] starting webcam stream...")
    from imutils.video import VideoStream
    video_stream = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    print(f"[INFO] loading video file: {args['video']}")
    video_stream = FileVideoStream(args["video"]).start()
    time.sleep(1.0)

output_path = "/content/output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = None
(H, W) = (None, None)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = video_stream.read()
	if frame is None:
		print("[INFO] end of stream.")
		break
	frame = imutils.resize(frame, width=400)

	if writer is None:
		(H, W) = frame.shape[:2]
		writer = cv2.VideoWriter(output_path, fourcc, 20, (W, H))
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# write the output frame to the video file
	writer.write(frame)

# do a bit of cleanup
print("[INFO] Output saved as output.mp4")
if args["video"] is None:
	video_stream.stop()
else:
	video_stream.stream.release()

if writer is not None:
	writer.release()