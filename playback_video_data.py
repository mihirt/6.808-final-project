import base64
import json
import cv2
import time

cap = cv2.VideoCapture('output.avi')

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with open('video_data.json', 'r') as inputfile:
    timestamp_arr = json.load(inputfile)

assert(len(timestamp_arr) == total)

for frame in dict_frames:
    # Convert back to binary
    jpg_original = base64.b64decode(frame)

    im = cv2.imread(jpg_original)

    # Display the resulting frame
    cv2.imshow('preview',im)
