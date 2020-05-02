def convert_rssi_to_distance(rssi, n=2.5):
    return 10**((3000 - rssi) / (10 * n))
