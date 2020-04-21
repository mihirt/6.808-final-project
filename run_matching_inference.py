import base64
import json
import cv2
import time
from IPython import embed
from collections import defaultdict
import numpy as np

with open('video_data.json', 'r') as inputfile:
    timestamp_data = json.load(inputfile)

with open('bounding_boxes.json', 'r') as inputfile:
    bbox_data = json.load(inputfile)

with open('rfid_data.json', 'r') as inputfile:
    rfid_data = json.load(inputfile)

def round_down(num, divisor):
    return num - (num%divisor)

def round_up(num, divisor):
    return round_down(num+divisor, divisor)


# binning
BIN_SZ = 50

start = round_up(max(timestamp_data[0], rfid_data[0][3]), BIN_SZ)
end = round_down(min(timestamp_data[-1], rfid_data[-1][3]), BIN_SZ)

camera_bins = {}
rfid_bins = {}

while timestamp_data[0] < start:
    timestamp_data = timestamp_data[1:]
    bbox_data = bbox_data[1:]

## ASSUMING CHRONOLOGICAL ORDER OF RFID DATA
while rfid_data[0][3] < start:
    rfid_data = rfid_data[1:]

while start < end:
    temp_bin = []
    while len(timestamp_data) > 0 and timestamp_data[0] < start + BIN_SZ:
        temp_bin.append(bbox_data[0])
        timestamp_data = timestamp_data[1:]
        bbox_data = bbox_data[1:]
    camera_bins[start] = temp_bin

    temp_bin = []
    while len(rfid_data) > 0 and rfid_data[0][3] < start + BIN_SZ:
        temp_bin.append(rfid_data[0])
        rfid_data = rfid_data[1:]
    rfid_bins[start] = temp_bin

    start += BIN_SZ

def compress_camera_bin(bin_obj):
    def get_centroid(bbox):
        return np.array([bbox[0] + (bbox[2]/2), bbox[1] + (bbox[3]/2)])

    person_to_centroids = defaultdict(list) # id -> [centroid, centroid...] = [[x, y], [x2, y2]]

    for reading in bin_obj:
        for id, bbox in reading.items():
            person_to_centroids[id].append(get_centroid(bbox))

    person_to_avg_centroid = {}

    for id, centroid_arr in person_to_centroids.items():
        centroid_arr = np.array(centroid_arr)
        avg_centroid = np.mean(centroid_arr, axis=0)
        person_to_avg_centroid[id] = avg_centroid
    
    return person_to_avg_centroid
        
def compress_rfid_bin(bin_obj):
    rfid_to_rssi_readings = defaultdict(list)

    for reading in bin_obj:
        obj_id = reading[0]
        obj_rssi = reading[1]
        rfid_to_rssi_readings[obj_id].append(obj_rssi)

    rfid_to_avg_rssi = {}
    
    for id, rssi_arr in rfid_to_rssi_readings.items():
        rssi_arr = np.array(rssi_arr)
        avg_rssi = np.mean(rssi_arr)
        rfid_to_avg_rssi[id] = avg_rssi

    return rfid_to_avg_rssi

processed_camera_bins = {timestamp: compress_camera_bin(obj) for timestamp, obj in camera_bins.items()}
processed_rfid_bins = {timestamp: compress_rfid_bin(obj) for timestamp, obj in rfid_bins.items()}



embed()

