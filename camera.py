import cv2
import datetime
import time

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]
counter = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
current_milli_time = lambda: int(round(time.time() * 1000))

while (counter < 50):
    frame_exists, curr_frame = cap.read()
    # if frame_exists:
    timestamps.append(current_milli_time())
    # timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    # calc_timestamps.append(calc_timestamps[-1] + 1000 / fps)
    counter += 1
    # else:
    #     break
    # out.write(curr_frame)

cap.release()
print(timestamps)
# print(calc_timestamps)
# for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
#     print('Frame %d difference:' % i, abs(ts - cts))
