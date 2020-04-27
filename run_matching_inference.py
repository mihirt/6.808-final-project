import base64
import json
import cv2
import time
from IPython import embed
from collections import defaultdict
import numpy as np
from numpy import cov
from scipy.stats import pearsonr, spearmanr

base_directory = 'data_multiple_1/'

with open(base_directory + 'video_data.json', 'r') as inputfile:
    timestamp_data = json.load(inputfile)

with open(base_directory + 'bounding_boxes.json', 'r') as inputfile:
    bbox_data = json.load(inputfile)

with open(base_directory + 'rfid_data.json', 'r') as inputfile:
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


def create_delta_dict(processed_bins):
    ''' 
    Calculate % differences between centroid and RFID values.
    '''
    delta_dict = {}
    for i, t in enumerate(sorted(processed_camera_bins.keys())):
        if i == 0:
            continue
        timestep_deltas = {}
        for id, val in processed_bins[t].items(): 
            if id in processed_bins[t-BIN_SZ]:
                prev_val = processed_bins[t-BIN_SZ][id]
                timestep_deltas[id] = (val - prev_val) / (prev_val)
        delta_dict[i] = timestep_deltas
    return delta_dict

delta_camera = create_delta_dict(processed_camera_bins)
delta_rfid = create_delta_dict(processed_rfid_bins)

def get_timesteps_with_id(data, id):
    '''
    Get all timesteps associated with any specific id (from the data).
    '''
    timesteps_arr = []
    for key, val in data.items():
        if id in val:
            timesteps_arr.append(key)
    return set(timesteps_arr)


def compute_similarity(rfid_id, camera_id):
    '''
    Computes the covariance, Pearson and Spearman correlations between RFID and Camera data.
    '''
    # first, chop RFID to person data! 
    # or chop to both?
    timesteps_rfid = get_timesteps_with_id(delta_rfid, rfid_id)
    timesteps_camera = get_timesteps_with_id(delta_camera, camera_id)

    # grab intersecting timesteps?
    intersecting_timesteps = timesteps_rfid.intersection(timesteps_camera)

    rfid = [delta_rfid[i][rfid_id] for i in intersecting_timesteps]
    camera_x = [delta_camera[i][camera_id][0] for i in intersecting_timesteps]
    camera_y = [delta_camera[i][camera_id][1] for i in intersecting_timesteps]

    # returns (cov (x, rfid), cov (y, rfid), pearson(x, rfid), pearson(y,rfid), spearman (x, rfid), spearman(y, rfid))
    cov_x_rfid = cov(camera_x, rfid)[0][1]
    cov_y_rfid = cov(camera_y, rfid)[0][1]

    pearson_x_rfid, _ = pearsonr(camera_x, rfid)
    pearson_y_rfid, _ = pearsonr(camera_y, rfid)

    spearman_x_rfid, _ = spearmanr(camera_x, rfid)
    spearman_y_rfid, _ = spearmanr(camera_y, rfid)

    return np.array([cov_x_rfid, cov_y_rfid, pearson_x_rfid, pearson_y_rfid, spearman_x_rfid, spearman_y_rfid])

def get_unique_ids(processed_bins):
    '''
    Get all unique ids (either camera or RFIDs).
    '''
    id_arr = []
    for bucket_obj in processed_bins.values():
        id_arr += bucket_obj.keys()
    return set(id_arr)

rfid_ids = get_unique_ids(delta_rfid)
camera_ids = get_unique_ids(delta_camera)

matching = {}

for rfid_id in rfid_ids:
    max_object = None
    max_object_val = 0.0
    
    for camera_id in camera_ids:
        similarity_arr = compute_similarity(rfid_id, camera_id)
        abs_similarity = np.absolute(similarity_arr)
        
        # multiply by weights
        weighted_similarity = np.multiply(abs_similarity, [1./6]*6)

        weighted_sum = np.sum(weighted_similarity)

        if weighted_sum > max_object_val:
            max_object = camera_id
            max_object_val = weighted_sum
        
        print('SIMILARITY between {} and {} = {} -> value of {}'.format(rfid_id, camera_id, similarity_arr, weighted_sum))


    matching[rfid_id] = max_object


# # NOTE: the matching datastructure contains the final inferred matchings. 
## TODO: integrate below delta methods with above inference procedures. 
embed()


def compute_delta_distance(start_bin_num, camerar_bins, bin_size = BIN_SZ, window_size = 10):
    '''Returns magnitude and direction of change for each obj'''
    end_bin_num = bin_size*window_size

    start_bin = camera_bins[start_bin_num] #dictionary with object id => np array of coords
    end_bin = camera_bins[end_bin_num + start_bin_num]

    output = []
    for obj in start_bin:
        if obj in end_bin:
            euc_dist = np.linalg.norm(start_bin[obj]-end_bin[obj])
            if np.linalg.norm(start_bin[obj]) > np.linalg.norm(end_bin[obj]):
                euc_dist = -1*euc_dist #sign indicates direction of change in mag
            output.append((obj, euc_dist))
    return output

def compute_delta_rfid(start_bin_num, rfid_bins, bin_size = 50, window_size = 10):
    '''Returns magnitude and direction of change, maybe output should be sorted in
    ascending order or somethin?'''
    end_bin_num = bin_size*window_size

    start_bin = rfid_bins[start_bin_num] #list of lists
    end_bin = rfid_bins[end_bin_num + start_bin_num]

    output = []
    for tag in start_bin:
        if tag in end_bin:
            delta = end_bin[tag] - start_bin[tag]
            output.append((tag, delta))
    return output

print(compute_delta_distance(1587490372400.0, processed_camera_bins))

