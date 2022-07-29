import kml2geojson
import json
jsonDict = kml2geojson.main.convert('CP_14Jal_v8.kml')[0]

with open('JALISCO.geojson', 'w') as fp:
    json.dump(jsonDict, fp)