import base64
import json
import cv2
import time

with open('video_data.json', 'r') as inputfile:
    dict_frames = json.load(inputfile)

for frame in dict_frames:
    # Convert back to binary
    jpg_original = base64.b64decode(frame)

    im = cv2.imread(jpg_original)

    # Display the resulting frame
    cv2.imshow('preview',im)
