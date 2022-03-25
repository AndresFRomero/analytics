import kml2geojson
import json
jsonDict = kml2geojson.main.convert('CP_15Mex_v8.kml')[0]

with open('MEXICO.geojson', 'w') as fp:
    json.dump(jsonDict, fp)