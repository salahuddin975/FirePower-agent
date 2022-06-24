import copy

import numpy as np


class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.rank = [1 for i in range(n)]

    def find(self, x):
        if x == self.root[x]:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.root[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.root[root_x] = root_y
            else:
                self.root[root_y] = root_x
                self.rank[root_x] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)


class ConnectedComponents:
    def __init__(self, generators):
        self.num_of_bus = 24
        self.num_of_branch = 34
        self.generators = generators
        self.reset()

    def reset(self):
        self._branch_status = np.ones(34)
        self._branches = [(0, 1),(0, 2),(0, 4),(1, 3),(1, 5),(2, 8),(2, 23),(3, 8),(4, 9),(5, 9),(6, 7),(7, 8),(7, 9),(8, 10),
                          (8, 11),(9, 10),(9, 11),(10, 12),(10, 13),(11, 12),(11, 22),(12, 22),(13, 15),(14, 15),(14, 20),
                          (14, 23),(15, 16),(15, 18),(16, 17),(16, 21),(17, 20),(18, 19),(19, 22),(20, 21)]
        self._find_all_connected_components()

    def _find_all_connected_components(self):
        self.connected_components = []
        self.union_find = UnionFind(self.num_of_bus)

        for branch in self._branches:
            if branch:
                self.union_find.union(branch[0], branch[1])

        for i in range(self.num_of_bus):
            flag = False
            for j in range(len(self.connected_components)):
                if self.union_find.connected(self.connected_components[j][0], i):
                    self.connected_components[j].append(i)
                    flag = True
                    break
            if flag == False:
                component = [i]
                self.connected_components.append(component)

        # print("all_connected_component: ", self.connected_components)

    def _remove_connected_components_if_no_active_generator(self, generator_output):
        if len(self.connected_components) == 1:
            return

        active_generators = []
        for i in self.generators.get_generators():
            if generator_output[i]:
                active_generators.append(i)

        for connected_component in reversed(self.connected_components):
            flag = False
            for i in connected_component:
                if i in active_generators:
                    flag = True
                    break
            if flag == False:
                self.connected_components.remove(connected_component)

        # print("active_generator_connected_components:", self.connected_components)

    def update_connected_components(self, state):
        branch_status = state["branch_status"]
        generator_output = state["generator_injection"]

        if (self._branch_status != branch_status).any():
            self._branch_status = copy.deepcopy(branch_status)
            for i, val in enumerate(self._branch_status):
                if val == 0:
                    self._branches[i] = 0
            self._find_all_connected_components()
            self._remove_connected_components_if_no_active_generator(generator_output)

    def get_connected_components(self):
        return self.connected_components