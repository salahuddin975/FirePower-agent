from PIL import Image, ImageDraw, ImageFont, ImageColor
import json
import random

PPC = 6  # pixel per cell
LINE_WIDTH = PPC // 3
FONT_SIZE = PPC * 6
FONT_STROKE_WIDTH = FONT_SIZE // 16


class Visualizer:
    def __init__(self, conf_file):
        with open(conf_file) as fin:
            self.conf_data = json.load(fin)

        self.branches = []
        for i in  self.conf_data['branches']:
            if i not in self.branches:
                self.branches.append(i)

    def draw_map(self, episode, step, fire_cells, burnt_cells, bus_status, branch_status, generation):
        image = Image.new('RGB', (self.conf_data['cols'], self.conf_data['rows']), ImageColor.getrgb('darkgreen'))

        for cell in self.conf_data['fuel_type']:
            assert (cell[2] == 0)
            image.putpixel((cell[0], cell[1]), ImageColor.getrgb('midnightblue'))

        for fire_cell in fire_cells:
            image.putpixel(fire_cell, ImageColor.getrgb('crimson'))

        for burnt_cell in burnt_cells:
            image.putpixel(burnt_cell, ImageColor.getrgb('brown'))

        draw = ImageDraw.Draw(image)

        buses = self.conf_data['bus_ids']
        for branch in self.conf_data['branches']:
            xy_from = buses[branch[0]][1:3]
            xy_to = buses[branch[1]][1:3]
            color = ImageColor.getrgb('gold') if branch_status[(branch[0], branch[1])] else ImageColor.getrgb('black')
            draw.line(xy_from + xy_to, fill=color)

        for bus in self.conf_data['bus_ids']:
            color = ImageColor.getrgb('gold') if bus_status[bus[0]] else ImageColor.getrgb('black')
            image.putpixel((bus[1], bus[2]), color)

        image = image.resize((self.conf_data['cols'] * PPC, self.conf_data['rows'] * PPC), Image.NEAREST)

        font = ImageFont.load_default()  # ImageFont.truetype("FreeSansBold.ttf", 24)
        # font = ImageFont.truetype("FreeSansBold.ttf", FONT_SIZE)

        draw = ImageDraw.Draw(image)
        draw.text((FONT_SIZE // 2, FONT_SIZE // 2),
                  f"Episode: {episode}   Step: {step}   Generation: {sum((generation[bus] for bus in generation)):.1f}",
                  font=font, fill=ImageColor.getrgb('white'), stroke_width=FONT_STROKE_WIDTH,
                  stroke_fill=ImageColor.getrgb('black'))

        for bus in self.conf_data['bus_ids']:
            x = bus[1]
            y = bus[2]
            if bus[0] in generation:
                text = f"{bus[0]}: {generation[bus[0]]:.1f}"
            else:
                text = f"{bus[0]}"
            if bus_status[bus[0]]:
                color = ImageColor.getrgb('white')
                stroke_color = ImageColor.getrgb('black')
            else:
                color = ImageColor.getrgb('black')
                stroke_color = ImageColor.getrgb('white')
            draw.rectangle((x * PPC - LINE_WIDTH, y * PPC - LINE_WIDTH, (x + 1) * PPC + LINE_WIDTH // 2,
                            (y + 1) * PPC + LINE_WIDTH // 2), outline=color, width=LINE_WIDTH)
            draw.text((x * PPC + FONT_SIZE // 2, y * PPC + FONT_SIZE // 2), text, font=font, fill=color,
                      stroke_width=FONT_STROKE_WIDTH, stroke_fill=stroke_color)

        return image


if __name__ == "__main__":
    # just for testing
    grid_size = 350
    conf_file = "configurations/configuration.json"
    with open(conf_file) as fin:
        data = json.load(fin)

    visualizer = Visualizer(conf_file)

    fire_cells = set([(random.randrange(0, grid_size), random.randrange(0, grid_size))])
    burnt_cells = set()
    bus_status = {bus[0]: random.random() < 0.7 for bus in data['bus_ids']}
    branch_status = {(branch[0], branch[1]): random.random() < 0.7 for branch in data['branches']}
    generation = {bus[0]: random.random() * 10.0 for bus in data['bus_ids'] if random.random() < 0.5}

    images = []

    for step in range(200):
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
        print("bus_status: ", bus_status)
        print("branch_status: ", branch_status)
        print("generation: ", generation)
        image = visualizer.draw_map(0, step, fire_cells, burnt_cells, bus_status, branch_status, generation)
        if step % 50 == 0:
            image.save(f"map_{step}.png")
        images.append(image)

    # images[0].save("map.gif", save_all=True, append_images=images[1:], loop=True)