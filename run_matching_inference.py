import base64
import json
import cv2
import time
from IPython import embed
from collections import defaultdict
import numpy as np
from numpy import cov
from scipy.stats import pearsonr, spearmanr
from rssi_to_distance import convert_rssi_to_distance

base_directory = 'large_space_2_box_fixed/'
NUM_FRAMES = 300 #must correspond to number specified in demo.py

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

timestamp_data = timestamp_data[:NUM_FRAMES] #shorten based on length of bbox data


# binning
BIN_SZ = 50

start = round_up(max(timestamp_data[0], rfid_data[0][3]), BIN_SZ)
end = round_down(min(timestamp_data[-1], rfid_data[-1][3]), BIN_SZ)
absolute_start = start
absolute_end = end

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

##################################
#Window Analysis Methods
def movingaverage(values, window):
    '''Computes Moving average using an average of window# of terms.'''
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'same')
    return sma

def create_deltas(array):
    '''takes array of size n. returns array of size n-1 corresponding to the delta
    changes between entries.'''
    out = [0]*(len(array)-1)
    for i in range(1, len(array)):
        out[i-1] = (array[i]-array[i-1])/array[i-1]
    return out

def compute_window_similarity(camera_x, camera_y, rfid):
    '''takes array of delta camera x-vals, array of camera y-vals,
    and array of delta rfid readings and computes covariances.'''
    cov_x_rfid = cov(camera_x, rfid)[0][1]
    cov_y_rfid = cov(camera_y, rfid)[0][1]

    pearson_x_rfid, _ = pearsonr(camera_x, rfid)
    pearson_y_rfid, _ = pearsonr(camera_y, rfid)

    spearman_x_rfid, _ = spearmanr(camera_x, rfid)
    spearman_y_rfid, _ = spearmanr(camera_y, rfid)

    return np.array([cov_x_rfid, cov_y_rfid, pearson_x_rfid, pearson_y_rfid, spearman_x_rfid, spearman_y_rfid])


def matching_in_window(bins_of_interest, people_of_interest, objects_of_interest):
    '''compute deltas for window  and use delta to compute covaraince'''
    # returns (cov (x, rfid), cov (y, rfid), pearson(x, rfid), pearson(y,rfid), spearman (x, rfid), spearman(y, rfid))

    matching = {}

    for rfid_id in objects_of_interest.keys():
        delta_rfid = create_deltas(objects_of_interest[rfid_id])
        max_object = None
        max_object_val = 0.0

        for camera_id in people_of_interest.keys():
            delta_camera_x = create_deltas(people_of_interest[camera_id][0])
            delta_camera_y = create_deltas(people_of_interest[camera_id][1])
            similarity_arr = compute_window_similarity(delta_camera_x, delta_camera_y, delta_rfid)
            abs_similarity = np.absolute(similarity_arr)

            # multiply by weights
            weighted_similarity = np.multiply(abs_similarity, [1./6]*6)

            weighted_sum = np.sum(weighted_similarity)

            if weighted_sum > max_object_val:
                max_object = camera_id
                max_object_val = weighted_sum

            print('SIMILARITY between {} and {} = {} -> value of {}'.format(rfid_id, camera_id, similarity_arr, weighted_sum))


        matching[rfid_id] = max_object
    return matching

def get_smooth_data_for_window(camera_data_dict, rfid_data_dict, start, window_size, bin_size):
    '''FOR GIVEN WINDOW
    For each person smooth x data, smooth y data,
    (even smooth depth data if we get it).
    For each tag, smooth rfid data'''

    bins_of_interest = []
    people_in_window = set()
    objects_in_window = set()

    #Determine bins in window
    end = min(int(start + window_size*bin_size), int(max(camera_data_dict.keys())))

    for i in range(int(start), end, bin_size):
        bins_of_interest.append(float(i))

    #Determine all people and objects that are detected at some point in window
    for b in bins_of_interest:
        for p in camera_data_dict[b].keys():
            people_in_window.add(p)
        for o in rfid_data_dict[b].keys():
            objects_in_window.add(o)

    #Initialize the data storage for each person and object in the window
    people_of_interest = {}
    objects_of_interest ={}
    for p in people_in_window: #init x,y for each bin
        people_of_interest[p] = ([-0.5]*window_size, [-0.5]*window_size)
    for o in objects_in_window: #init rssi for each bin
        objects_of_interest[o] = [-0.5]*window_size

    #For each person and each object, iterate through bins and store data
    for p in people_of_interest:
        i = 0
        for b in bins_of_interest:
            try:
                x,y = camera_data_dict[b][p]
                people_of_interest[p][0][i] = x
                people_of_interest[p][1][i] = y
            except:
                pass
            i+=1

    for o in objects_of_interest:
        i = 0
        for b in bins_of_interest:
            try:
                rssi = convert_rssi_to_distance(rfid_data_dict[b][o]) #convert to true distance
                objects_of_interest[o][i] = rssi
            except:
                pass
            i+=1

    # Display Data Before Smoothing
    # print(objects_of_interest)
    # print(people_of_interest)

    # SMOOTH DATA
    avg_window = 3
    for p in people_of_interest:
        xvalues = people_of_interest[p][0]
        yvalues = people_of_interest[p][1]
        people_of_interest[p] = (movingaverage(xvalues, avg_window), movingaverage(yvalues, avg_window))

    for o in objects_of_interest:
        values = objects_of_interest[o]
        objects_of_interest[o] = movingaverage(values, avg_window)

    return bins_of_interest, people_of_interest, objects_of_interest

print("runnnnnn")
print(absolute_start, absolute_end)
for t in range(int(absolute_start), int(absolute_end), 50):
    print(t)
    timestep = float(t)
    b,p,o = get_smooth_data_for_window(processed_camera_bins, processed_rfid_bins, timestep, 10, 50)
    print(matching_in_window(b, p, o))

########################################
#Full Video Analysis Methods

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

embed()
