import time
import numpy as np
import cv2 as cv
# import cPickle as pickle
import mercury

cap = cv.VideoCapture(0)
reader = mercury.Reader("tmr:///dev/cu.usbmodem146101")

reader.set_region("NA")
reader.set_read_plan([1], "GEN2")

timeStampedImages = []
newVal = 0


def readCamera(tag):
    global newVal
    print(tag.rssi)
    newVal = tag.rssi


max = reader.get_power_range()[1]
print(reader.set_read_powers([(1, 2000)]))

reader.start_reading(readCamera)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # print(tag.rssi)
    frame = cv.flip(frame, 0)
    # write the flipped frame
    # out.write(frame)
    font = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (0, 255, 255)
    lineType = 2

    cv.putText(frame, str(newVal), bottomLeftCornerOfText, font, fontScale,
               fontColor, lineType)
    timeStampedImages.append((frame, time.time()))
    img_str = cv2.imencode('.jpg', frame)[1].tostring()

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        reader.stop_reading()

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
