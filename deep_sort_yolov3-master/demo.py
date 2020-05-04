from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import json
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

BASE_DIR = '/Users/ericpence/Desktop/808_proj/env1/'
FRAME_COUNT = 300

def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

   # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    video_capture = cv2.VideoCapture(BASE_DIR+'6.808-final-project/large_space_2_box_fixed/video.avi')
    # timestamps_cv_lib = []
    # timestamps_calc = []

    bboxes_per_frame = []


    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0

    frame_count = FRAME_COUNT #init frame_count

    while frame_count > 0:
        print(frame_count)
        frame_count -= 1
        ret, frame = video_capture.read()  # frame shape 640*480*3
        track2bbox = {}
        if ret != True:
            break

        # measured_fps = video_capture.get(cv2.CAP_PROP_FPS)
        # timest = float(frame_count)*1000.0/measured_fps #timestamp in ms
        # timestamps_cv_lib.append(timest)
        # frame_count+=1 #using cv lib

        t1 = time.time() #just using time lib
        # timestamps_calc.append(t1) #USE TIME LIBRARY TO GET TIME STAMP

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            track2bbox[track.track_id] = bbox.tolist() #keep dict matching person to box

        bboxes_per_frame.append(track2bbox) #store in chronological order in list

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        #cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    # print(timestamps_cv_lib)
    # print(timestamps_calc)
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

    with open(BASE_DIR+'6.808-final-project/large_space_2_box_fixed/bounding_boxes.json', 'w') as outfile:
        json.dump(bboxes_per_frame, outfile)  #write dictionary to json

if __name__ == '__main__':
    main(YOLO())
