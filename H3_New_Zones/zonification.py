# import kml2geojson
# import json
# jsonDict = kml2geojson.main.convert('mc_comunas.kmz')[0]

# with open('comunas_cali.geojson', 'w') as fp:
#     json.dump(jsonDict, fp)


import geopandas

shp_file = geopandas.read_file('mc_comunas.shp')

shp_file.to_file('mc_comunas.geojson', driver='GeoJSON')
