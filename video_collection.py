import base64
import json
import cv2
import time

stamp_to_image = {} #write to json later

# Open the device at the ID 0
cap = cv2.VideoCapture(0)

#Check whether user selected camera is opened successfully.

if not (cap.isOpened()):
    print('Could not open video device')

#To set the resolution, Necessary??
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 300 #300 frames is 10 seconds of video
while count > 0:
    count -= 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    stamp = time.time()*1000.0 # timestamp in ms
    retval, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    stamp_to_image[stamp] = jpg_as_text #key = stamp, value = image

    # Display the resulting frame
    cv2.imshow('preview',frame)

    #Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()

with open('video_data.json', 'w') as outfile:
    json.dump(stamp_to_image, outfile) #write dictionary to json
