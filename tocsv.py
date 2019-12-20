
import json
import csv


data = json.load(open('testdata.json', 'r'))

with open('testdata.csv', mode='w') as data_file:
    writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

    for datum in data:
        writer.writerow([datum['time'], datum['open'], datum['high'], datum['low'], datum['close'], datum['volume']])