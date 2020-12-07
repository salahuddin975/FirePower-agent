import os
import git


class Parameters:
    def __init__(self, base_path, model_version):
        self._base_dir = base_path
        self._model_version = model_version
        self._file_name = os.path.join(base_path, "parameters")
        self._create_dir()

        self._agent_git_repo = git.Repo(search_parent_directories=True)
        self._simulator_git_repo = git.Repo(path="./../gym-firepower/")

        # ------------ agent.py -------------
        self.gamma = 0.9                    # False
        self.hidden_layer1 = 512            # False
        self.hidden_layer2 = 128            # False
        self.obs_fields = "bus_input, branch_input, fire_distance_input, gen_inj_input, load_demand_input, theta_input, line_flow_input"           # False

        # ------------ main training loop -----------
        self.generator_max_output = False           # True
        self.noise_rate = 0.5                       # False
        self.test_after_episodes = 20               # True

        # ----------- commit history ---------------
        self.agent_branch = self._agent_git_repo.active_branch.name
        self.agent_commit_number = self._agent_git_repo.head.object.hexsha
        self.simulator_branch = self._simulator_git_repo.active_branch.name
        self.simulator_commit_number = self._simulator_git_repo.head.object.hexsha

        self._initialize()

    def _initialize(self):
        self.parameters = \
            "------------ Agent NN ---------- \n" + \
            "gamma: " + str(self.gamma) + "\n" + \
            "first hidden layer: " + str(self.hidden_layer1) + "\n" + \
            "second hidden layer: " + str(self.hidden_layer2) + "\n" + \
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
            "simulator commit number: " + self.simulator_commit_number + "\n"

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
