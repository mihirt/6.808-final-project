import cv2
import numpy as np
import sys
import test_simple
from PIL import Image
from test_simple import evaluate_single
import json


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
def getCenterCoord(coord):
    return (coord[0] + (coord[2] / 2), coord[1] + (coord[3] / 2))


def normalize(coords, size):
    return ((coords[0] / size[0], coords[1] / size[1]))


video_file = sys.argv[1]
bbox_file = sys.argv[2]
cap = cv2.VideoCapture(video_file)
bbox = open(bbox_file)
bbox_list = json.load(bbox)
print(bbox_file)
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
depth_list = []
counter = 0
# cap.set(2, 0.06)
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(ret)
    if ret == True:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sz = pil_image.size
        current_bboxes = bbox_list[counter]
        print(counter)
        depth_values, image = evaluate_single(pil_image)
        numpy_image = np.array(image)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        dict = {}
        for key in current_bboxes:
            center = getCenterCoord(current_bboxes[key])
            center = normalize(center, sz)
            newLoc = (int(center[0] * 640) - 1, int(center[1] * 191) - 2)
            print(newLoc)
            depth = depth_values[0][0][newLoc[1]][640 - newLoc[0]]
            print(depth)
            cv2.circle(opencv_image, (newLoc[1], 640 - newLoc[0]), 10,
                       (0, 0, 255), 5)
            dict[key] = str(depth)
        depth_list.append(dict)

        # cv2.imwrite("bob.jpg", opencv_image)
        # print(key, normalize(center, sz))

        counter += 1
        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        #
        # # Press Q on keyboard to  exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
with open('depths.json', 'w') as f:
    json.dump(depth_list, f)
# Closes all the frames
# cv2.destroyAllWindows()
