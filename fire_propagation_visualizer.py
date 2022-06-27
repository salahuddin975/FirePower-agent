import copy
import json
import random
from PIL import Image, ImageDraw, ImageFont, ImageColor

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


class Visualizer:
    def __init__(self, generators, simulator_resources, conf_file):
        self._generators = generators
        self.simulator_resources = simulator_resources

        with open(conf_file) as fin:
            self.conf_data = json.load(fin)

        self.branches = []
        for i in  self.conf_data['branches']:
            if i not in self.branches:
                self.branches.append(i)

    def draw_map(self, episode, step, cells_info, state):
        burning_cells = cells_info[0]
        burnt_cells = cells_info[1]
        bus_status = copy.deepcopy(state["bus_status"])
        branch_status = copy.deepcopy(state["branch_status"])
        generation = copy.deepcopy(state["generator_injection"])
        load_demand = copy.deepcopy(state["load_demand"])
        # branch_status = self._check_network_violations_branch(bus_status, branch_status)

        image = Image.new('RGB', (self.conf_data['cols'], self.conf_data['rows']))

        for cell in self.conf_data["fuel_amt"]:
            for type in CELL_TYPES.values():
                if cell[2] == type[1]:
                    image.putpixel((cell[0], cell[1]), type[0])

        for fire_cell in burning_cells:
            image.putpixel((fire_cell[1], fire_cell[0]), COLORS["FIRE_BURNING"])

        for burnt_cell in burnt_cells:
            image.putpixel((burnt_cell[1], burnt_cell[0]), COLORS["FIRE_BURNT"])

        draw = ImageDraw.Draw(image)

        buses = self.conf_data['bus_ids']
        for i, branch in enumerate(self.branches):
            xy_from = buses[branch[0]][1:3]
            xy_to = buses[branch[1]][1:3]
            color = COLORS["ACTIVE_BRANCH"] if branch_status[i] else COLORS["INACTIVE_BRANCH"]
            draw.line(xy_from + xy_to, fill=color)

        for bus in self.conf_data['bus_ids']:
            color = ImageColor.getrgb('gold') if bus_status[bus[0]] else ImageColor.getrgb('black')
            image.putpixel((bus[1], bus[2]), color)

        image = image.resize((self.conf_data['cols'] * PPC, self.conf_data['rows'] * PPC), Image.NEAREST)

        font = ImageFont.truetype("FreeSansBold.ttf", FONT_SIZE)
        draw = ImageDraw.Draw(image)
        draw.text((FONT_SIZE // 2, FONT_SIZE // 2),
                  f"Episode: {episode}   Step: {step}   Generation: {sum(generation):.1f}",
                  font=font, fill=ImageColor.getrgb('white'), stroke_width=FONT_STROKE_WIDTH,
                  stroke_fill=ImageColor.getrgb('black'))

        for bus in self.conf_data['bus_ids']:
            x = bus[1]
            y = bus[2]
            # if generation[bus[0]]:
            if bus[0] in self._generators.get_generators():
                text = f"{bus[0]}: {load_demand[bus[0]]:.2f}, {generation[bus[0]]:.2f}"
                color = ImageColor.getrgb('white')
            else:
                text = f"{bus[0]}: {load_demand[bus[0]]:.2f}"
                color = ImageColor.getrgb('gold')
            if bus_status[bus[0]]:
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
    bus_status = np.ones(24)  # {bus[0]: random.random() < 0.7 for bus in data['bus_ids']}
    branch_status = np.ones(34) # {(branch[0], branch[1]): random.random() < 0.7 for branch in data['branches']}
    generation = np.ones(24) # {bus[0]: random.random() * 10.0 for bus in data['bus_ids'] if random.random() < 0.5}

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

# scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/fire_propagation_0_\*.png  .

# ffmpeg -framerate 30 -pattern_type sequence -i 'fire_propagation_0_%d.png' -c:v libx264 -pix_fmt yuv420p episode_0.mp4
# for i in `seq 0 5`; do ffmpeg -framerate 30 -pattern_type sequence -i fire_propagation_${i}_%d.png -c:v libx264 -pix_fmt yuv420p episode_${i}.mp4 ; done
