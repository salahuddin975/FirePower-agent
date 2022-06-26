from PIL import Image, ImageDraw, ImageFont, ImageColor
import json
import random

PPC = 6  # pixel per cell
LINE_WIDTH = PPC // 3
FONT_SIZE = PPC * 6
FONT_STROKE_WIDTH = FONT_SIZE // 24

CELL_TYPES = {
    'LAND': ((148, 210, 165), 90),
    'WATER': ((138, 180, 248), 0),
    'DESERT': ((245, 240, 228), 20),
    'FOREST': ((143, 203, 160), 100),
    'LOW_VEGETATION': ((187, 226, 198), 80),
    'DEEP_DESERT': ((255, 255, 255), 10),
}

COLORS = {
    'ACTIVE_BRANCH': ImageColor.getrgb('black'),
    'INACTIVE_BRANCH': ImageColor.getrgb('red'),
    'ACTIVE_BUS': ImageColor.getrgb('black'),
    'INACTIVE_BUS': ImageColor.getrgb('brown'),
    'ACTIVE_TEXT': ImageColor.getrgb('black'),
    'ACTIVE_TEXT_OUTLINE': ImageColor.getrgb('white'),
    'INACTIVE_TEXT': ImageColor.getrgb('brown'),
    'INACTIVE_TEXT_OUTLINE': ImageColor.getrgb('white'),
    'FIRE_BURNING': ImageColor.getrgb('orange'),
    'FIRE_BURNT': ImageColor.getrgb('brown')
}

with open("configuration.json") as fin:
    data = json.load(fin)

with open("../fuel_type.json") as fin:
    fuel_type = json.load(fin)


def draw_map(episode, step, fire_cells, burnt_cells, bus_status, branch_status, generation):
    image = Image.new('RGB', (data['cols'], data['rows']))

    for cell in fuel_type:
        for type in CELL_TYPES.values():
            if cell[2] == type[1]:
                image.putpixel((cell[0], cell[1]), type[0])

    for fire_cell in fire_cells:
        image.putpixel(fire_cell, COLORS['FIRE_BURNING'])

    for burnt_cell in burnt_cells:
        image.putpixel(burnt_cell, COLORS['FIRE_BURNT'])

    draw = ImageDraw.Draw(image)

    buses = data['bus_ids']
    for branch in data['branches']:
        xy_from = buses[branch[0]][1:3]
        xy_to = buses[branch[1]][1:3]
        color = COLORS['ACTIVE_BRANCH'] if branch_status[(branch[0], branch[1])] else COLORS['INACTIVE_BRANCH']
        draw.line(xy_from + xy_to, fill=color)

    for bus in data['bus_ids']:
        color = COLORS['ACTIVE_BUS'] if bus_status[bus[0]] else COLORS['INACTIVE_BUS']
        image.putpixel((bus[1], bus[2]), color)

    image = image.resize((data['cols'] * PPC, data['rows'] * PPC), Image.NEAREST)

    font = ImageFont.truetype("FreeSansBold.ttf", FONT_SIZE)
    draw = ImageDraw.Draw(image)
    draw.text((FONT_SIZE // 2, FONT_SIZE // 2),
              f"Episode: {episode}   Step: {step}   Generation: {sum((generation[bus] for bus in generation)):.1f}",
              font=font, fill=COLORS['ACTIVE_TEXT'], stroke_width=FONT_STROKE_WIDTH,
              stroke_fill=COLORS['ACTIVE_TEXT_OUTLINE'])

    for bus in data['bus_ids']:
        x = bus[1]
        y = bus[2]
        if bus[0] in generation:
            text = f"{bus[0]}: {generation[bus[0]]:.1f}"
        else:
            text = f"{bus[0]}"
        if bus_status[bus[0]]:
            color = COLORS['ACTIVE_TEXT']
            stroke_color = COLORS['ACTIVE_TEXT_OUTLINE']
        else:
            color = COLORS['INACTIVE_TEXT']
            stroke_color = COLORS['INACTIVE_TEXT_OUTLINE']
        draw.rectangle((x * PPC - LINE_WIDTH, y * PPC - LINE_WIDTH, (x + 1) * PPC + LINE_WIDTH // 2,
                        (y + 1) * PPC + LINE_WIDTH // 2), outline=color, width=LINE_WIDTH)
        draw.text((x * PPC + FONT_SIZE // 2, y * PPC + FONT_SIZE // 2), text, font=font, fill=color,
                  stroke_width=FONT_STROKE_WIDTH, stroke_fill=stroke_color)

    return image


# just for testing
fire_cells = set([(random.randrange(0, data['cols']), random.randrange(0, data['rows']))])
burnt_cells = set()
bus_status = {bus[0]: random.random() < 0.5 for bus in data['bus_ids']}
branch_status = {(branch[0], branch[1]): random.random() < 0.5 for branch in data['branches']}
generation = {bus[0]: random.random() * 10.0 for bus in data['bus_ids'] if random.random() < 0.5}

images = []

for step in range(51):
    # just for testing
    spread = set()
    for fire_cell in fire_cells:
        x = fire_cell[0]
        y = fire_cell[1]
        spread_x = random.randint(x - 1, x + 1)
        spread_y = random.randint(y - 1, y + 1)
        if (0 <= spread_x < data['cols']) and (0 <= spread_y < data['rows']):
            spread.add((spread_x, spread_y))
        if (random.random() < 0.1):
            burnt_cells.add(fire_cell)
        else:
            spread.add(fire_cell)
        spread.difference_update(burnt_cells)
    fire_cells = spread

    print(f"Drawing step {step}")
    image = draw_map(0, step, fire_cells, burnt_cells, bus_status, branch_status, generation)
    if step % 50 == 0:
        image.save(f"map_{step}.png")
    images.append(image)

# images[0].save("map.gif", save_all=True, append_images=images[1:], loop=True)