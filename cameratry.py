import time
import numpy as np
import cv2 as cv
# import cPickle as pickle
import mercury
from imutils.video import WebcamVideoStream
from imutils.video import FPS
cap = cv.VideoCapture(0)
try:
    reader = mercury.Reader("tmr:///dev/cu.usbmodem146101")
    reader.set_region("NA")
    reader.set_read_plan([1], "GEN2")
    max = reader.get_power_range()[1]
    print(reader.set_read_powers([(1, 3000)]))

    def readCamera(tag):
        global newVal
        print(tag.rssi)
        newVal = tag.rssi

    reader.start_reading(readCamera)
except Exception as e:
    print(e)

timeStampedImages = []
newVal = 0
count = 0
prevTime = time.time()
# cap = WebcamVideoStream(src=0).start()
f = FPS().start()
while count < 200:
    ret, frame = cap.read()
    if not True:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # print(tag.rssi)
    frame = cv.flip(frame, 0)
    if count % 30 == 0:
        print((time.time() - prevTime) * 1000)
        prevTime = time.time()
    # write the flipped frame
    # out.write(frame)
    font = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 5
    fontColor = (20, 0, 255)
    lineType = 4

    cv.putText(frame, str(newVal), bottomLeftCornerOfText, font, fontScale,
               fontColor, lineType)
    timeStampedImages.append((frame, time.time()))
    # img_str = cv.imencode('.jpg', frame)[1].tostring()
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        reader.stop_reading()
    count += 1
    f.update()
f.stop()
print(f.fps())

# while True:
#     pass

# Release everything if job is finished

# pickle.dump(timeStampedImages, outfile)
# outfile.close()
# out.release()

# infile = open(filename, 'rb')
# tVide = pickle.load(infile)
# infile.close()
#
# for item in tVide:
#     cv.imshow('frame', item[0])
