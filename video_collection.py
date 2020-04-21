import base64
import json
import cv2
import time
import argparse
from datetime import datetime
import os

def _main(args):
    stamps_arr = [] #write to json later

    base_folder = 'output/' + args.folder_name
    os.makedirs(base_folder)
    print(base_folder)

    # Open the device at the ID 0
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # For writing the file -> to the 'out' handler
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(base_folder + '/video.avi',fourcc, fps, (frame_width, frame_height))

    #Check whether user selected camera is opened successfully.

    if not (cap.isOpened()):
        raise Exception('Could not open video device')

    # #To set the resolution, Necessary??
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count = int(args.frame_count) #300 frames is 10 seconds of video
    while count > 0:
        count -= 1
        print(count)
        # Capture frame-by-frame
        ret, frame = cap.read()
        stamp = time.time()*1000.0 # timestamp in ms
        # retval, buffer = cv2.imencode('.jpg', frame)
        # jpg_as_text = base64.b64encode(buffer)
        # stamp_to_image[stamp] = jpg_as_text #key = stamp, value = image
        out.write(frame)
        stamps_arr.append(stamp)


        # Display the resulting frame
        cv2.imshow('preview',frame)

        #Waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(base_folder + '/video_data.json', 'w') as outfile:
        json.dump(stamps_arr, outfile) #write dictionary to json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record and save video input and timestamps')
    parser.add_argument('--folder_name', default='experiment_'+datetime.now().strftime("%H:%M:%S"),
                        help='folder name for output files')
    parser.add_argument('--frame_count', default=300,
                        help='number of frames to record')

    _main(parser.parse_args())
