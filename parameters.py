import os
import git
import json


class Parameters:
    def __init__(self, base_path, model_version, geo_file):
        self._base_dir = base_path
        self._model_version = model_version
        self._file_name = os.path.join(base_path, "parameters")
        self._create_dir()

        self._agent_git_repo = git.Repo(search_parent_directories=True)
        self._simulator_git_repo = git.Repo(path="./../gym-firepower/")

        # ------------ agent.py -------------
        self.gamma = 0.99                        # False
        self.hidden_layer = "512, 128"           # False
        self.obs_fields = "bus_input, branch_input, fire_distance_input, gen_inj_input, load_demand_input, theta_input, line_flow_input"           # False

        # ------------ main training loop -----------
        self.generator_max_output = False           # True
        self.noise_rate = 0.5                       # True
        self.test_after_episodes = 20               # True

        # ----------- commit history ---------------
        try:
            self.agent_branch = self._agent_git_repo.active_branch.name
        except:
            self.agent_branch = "detached head"
        self.agent_commit_number = self._agent_git_repo.head.object.hexsha
        try:
            self.simulator_branch = self._simulator_git_repo.active_branch.name
        except:
            self.simulator_branch = "detached head"
        self.simulator_commit_number = self._simulator_git_repo.head.object.hexsha

        # ----------- fire spread conf ---------------
        self._parse_geo_file(geo_file)

        self._initialize()

    def _initialize(self):
        self.parameters = \
            "------------ Agent NN ---------- \n" + \
            "gamma: " + str(self.gamma) + "\n" + \
            "hidden layers: " + self.hidden_layer + "\n" + \
            "observation fields: " + self.obs_fields + "\n" + \
            \
            "\n ------------ Main training loop --------- \n" + \
            "generator max output: " + str(self.generator_max_output)  + "\n" + \
            "noise_range: " + str(self.noise_rate)  + "\n" + \
            "test after every: " + str(self.test_after_episodes) + " episodes"  + "\n" + \
            \
            "\n ------------ Commit history --------- \n" + \
            "agent branch: " + self.agent_branch + "\n" + \
            "agent commit number: " + self.agent_commit_number + "\n" + \
            "simulator branch: " + self.simulator_branch + "\n" + \
            "simulator commit number: " + self.simulator_commit_number + "\n" + \
            \
            "\n ------------ fire spread conf --------- \n" + \
            "random_source: " + str(self._random_source) + "\n" + \
            "fixed_source: " + str(self._fixed_source) + "\n" + \
            "box: " + str(self._boxes) + "\n" + \
            "num of sources: " + str(self._num_sources) + "\n"

    def _parse_geo_file(self, geo_file):
        args = {"cols": 40, "rows": 40, "sources": [[5, 5]],
                "seed": 30, "random_source":False, "num_sources":1}

        with open(geo_file, 'r') as config_file:
            args.update(json.load(config_file))

        self._rows = int(args["rows"])
        self._cols = int(args["cols"])
        self._fixed_source = args["sources"]
        self._random_source = args["random_source"]
        self._boxes = args["boxes"]
        self._num_sources = args["num_sources"]

    def _create_dir(self):
        try:
            os.makedirs(self._base_dir)
        except OSError as error:
            print(error)

    def save_parameters(self):
        with open(f'{self._file_name}_v{self._model_version}.txt', 'w') as fd:
            fd.write(self.parameters)

    def print_parameters(self):
        print(self.parameters)
