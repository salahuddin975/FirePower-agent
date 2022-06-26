from PIL import Image
import json

COLUMNS = 350
ROWS = 350

CELL_TYPES = {
  'LAND': ((148, 210, 165), 90),
  'WATER': ((138, 180, 248), 0),
  'DESERT': ((245, 240, 228), 20),
  'FOREST': ((143, 203, 160), 100),
  'LOW_VEGETATION': ((187, 226, 198), 80),
  'DEEP_DESERT': ((255, 255, 255), 10),
  'ROAD': ((249, 173, 70), 20), # map roads to desert
}

with Image.open("WECC area - water cleaned - scaled.png").convert('RGB') as image:
  assert (image.size == (COLUMNS, ROWS))
  map = []
  for x in range(COLUMNS):
    for y in range(ROWS):
      color = image.getpixel((x, y))
      fuel_type = None
      color_distance = float('inf')
      for type in CELL_TYPES.values():
        distance = sum([(color[i] - type[0][i])**2 for i in range(3)])
        if distance < color_distance:
          color_distance = distance
          fuel_type = type[1]
      map.append([x, y, fuel_type])

with open('fuel_type.json', 'w') as fout:
  json.dump(map, fout)