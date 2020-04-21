import base64
import json
import cv2
import time
import argparse
from datetime import datetime
import os
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import mercury
import threading

reader = mercury.Reader("tmr:///dev/cu.usbmodem146101")
rssi_timestamps = []


def start_rfid():
    reader.set_region("NA")
    reader.set_read_plan([1], "GEN2")
    max = reader.get_power_range()[1]
    print(reader.set_read_powers([(1, 3000)]))

    def readCamera(tag):
        print(tag.rssi)
        encoded = tag.epc.decode('utf-8')
        # print(type(time.time()))
        rssi_timestamps.append(
            (encoded, tag.rssi, tag.phase, time.time() * 1000))

    reader.start_reading(readCamera)


def _main(args):
    stamps_arr = []  #write to json later

    base_folder = 'output/' + args.folder_name
    os.makedirs(base_folder)
    print(base_folder)
    flip = args.flip
    print(flip)
    # Open the device at the ID 0
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    # cap.set(cv2.CV_CAP_PROP_BUFFERSIZE, 60)
    # For writing the file -> to the 'out' handler
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(base_folder + '/video.avi', fourcc, fps,
                          (frame_width, frame_height))

    #Check whether user selected camera is opened successfully.

    if not (cap.isOpened()):
        raise Exception('Could not open video device')

    # #To set the resolution, Necessary??
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = FPS().start()
    cap1 = WebcamVideoStream(src=0).start()
    count = int(args.frame_count)  #300 frames is 10 seconds of video
    while count > 0:
        count -= 1
        # Capture frame-by-frame
        # ret, frame = cap.read()
        frame = cap1.read()
        if flip:
            frame = cv2.flip(frame, 0)
        stamp = time.time() * 1000.0  # timestamp in ms
        stamps_arr.append(stamp)

        # Display the resulting frame
        # cv2.imshow('preview', frame)
        out.write(frame)
        fps.update()
        #Waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    fps.stop()
    print(fps.fps())
    with open(base_folder + '/video_data.json', 'w') as outfile:
        json.dump(stamps_arr, outfile)  #write dictionary to json
    with open(base_folder + '/rfid_data.json', 'w') as outfile:
        json.dump(rssi_timestamps, outfile)  #write dictionary to json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Record and save video input and timestamps')
    parser.add_argument('--folder_name',
                        default='experiment_' +
                        datetime.now().strftime("%H:%M:%S"),
                        help='folder name for output files')
    parser.add_argument('--frame_count',
                        default=300,
                        help='number of frames to record')
    parser.add_argument('-flip', default=False)
    th = threading.Thread(target=_main, args=(parser.parse_args(), ))
    th.start()
    start_rfid()
    th.join()
    reader.stop_reading()
    # _main(parser.parse_args())
