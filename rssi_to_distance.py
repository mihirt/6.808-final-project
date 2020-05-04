def convert_rssi_to_distance(rssi, n=5):
    return 10**((30 - (rssi / 100)) / (10 * n))
