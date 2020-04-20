import mercury
import cv2
import datetime
reader = mercury.Reader("tmr:///dev/cu.usbmodem146101")

reader.set_region("NA")
reader.set_read_plan([1], "GEN2")
# tags = set(reader.read())
# rssi = [i.rssi for i in tags]
# print(rssi)


def stats_received(stats):
    print({"temp": stats.temperature})
    print({"antenna": stats.antenna})
    print({"protocol": stats.protocol})
    print({"frequency": stats.frequency})


# reader.enable_stats(stats_received)
tags_seen = []
reader.start_reading(lambda tag: tags_seen.append((tag.rssi, tag.epc)))
counter = 0
while counter < 10**8:
    counter += 1
reader.stop_reading()
