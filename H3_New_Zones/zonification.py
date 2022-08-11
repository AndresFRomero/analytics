# import kml2geojson
# import json
# jsonDict = kml2geojson.main.convert('SaoPaulo.kml')[0]

# with open('SaoPaulo.geojson', 'w') as fp:
#     json.dump(jsonDict, fp)


import geopandas

shp_file = geopandas.read_file('MGN_MPIO_POLITICO.shp')

shp_file.to_file('myshpfile.geojson', driver='GeoJSON')
