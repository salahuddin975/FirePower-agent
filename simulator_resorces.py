import numpy as np
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_bus import *
from pypower.loadcase import loadcase
from pypower.ext2int import ext2int


class Generators:
    def __init__(self, ppc, power_generation_preprocess_scale, ramp_frequency_in_hour):
        self.ppc = ppc
        self.power_generation_preprocess_scale = power_generation_preprocess_scale
        self.ramp_frequency_in_hour = ramp_frequency_in_hour

        self.reset()
        # self.print_info()

    def reset(self):
        self._generators = np.copy(self.ppc["gen"][:, GEN_BUS].astype("int"))
        self._num_generators = self._generators.size

        self._generators_min_output = np.copy(self.ppc["gen"][:, PMIN] / (self.power_generation_preprocess_scale * self.ppc["baseMVA"]))
        self._generators_max_output = np.copy(self.ppc["gen"][:, PMAX] / (self.power_generation_preprocess_scale * self.ppc["baseMVA"]))
        self._generators_max_ramp = np.copy(self.ppc["gen"][:, RAMP_10] / (self.power_generation_preprocess_scale * self.ppc["baseMVA"] * self.ramp_frequency_in_hour))
        self.remove_generator(13)

    def remove_generator(self, generator):
        index = -1
        for i in range(len(self._generators)):
            if self._generators[i] == generator:
                index = i

        if index != -1:
            self._num_generators -= 1
            self._generators = np.delete(self._generators, index)
            self._generators_min_output = np.delete(self._generators_min_output, index)
            self._generators_max_output = np.delete(self._generators_max_output, index)
            self._generators_max_ramp = np.delete(self._generators_max_ramp, index)

    def get_generators(self):
        return self._generators

    def get_num_generators(self):
        return self._num_generators

    def get_min_outputs(self):
        return self._generators_min_output

    def get_max_outputs(self):
        return self._generators_max_output

    def get_max_ramps(self):
        return self._generators_max_ramp

    def set_max_outputs(self, max_output):
        self._generators_max_output = np.copy(max_output[self._generators])

    def set_zero_for_generator(self, i):
        self._generators_min_output[i] = 0
        self._generators_max_output[i] = 0
        self._generators_max_ramp[i] = 0

    def set_max_output(self, bus_id, max_output):
        index = -1
        for i in range(len(self._generators)):
            if self._generators[i] == bus_id:
                index = i
        self._generators_max_output[index] = max_output

    def set_min_output(self, bus_id, min_output):
        index = -1
        for i in range(len(self._generators)):
            if self._generators[i] == bus_id:
                index = i
        self._generators_min_output[index] = min_output

    def set_max_ramp(self, bus_id, max_ramp):
        index = -1
        for i in range(len(self._generators)):
            if self._generators[i] == bus_id:
                index = i
        self._generators_max_ramp[index] = max_ramp

    def print_info(self):
        print ("generators: ", self._generators)
        print ("generators min output: ", self._generators_min_output)
        print ("generators max output: ", self._generators_max_output)
        print ("generators max ramp: ", self._generators_max_ramp)


class SimulatorResources():
    def __init__(self, power_file_path, power_generation_preprocess_scale):
        self._ppc = loadcase(power_file_path)
        self._merge_generators()
        self._merge_branches()
        self.ppc = ext2int(self._ppc)
        self._power_generation_preprocess_scale = power_generation_preprocess_scale

    def _merge_generators(self):
        ppc_gen_trim = []
        temp = self._ppc["gen"][0, :]
        ptr = 0
        ptr1 = 1
        while ptr1 < self._ppc["gen"].shape[0]:
            if self._ppc["gen"][ptr, GEN_BUS] == self._ppc["gen"][ptr1, GEN_BUS]:
                temp[PG:QMIN+1] += self._ppc["gen"][ptr1, PG:QMIN+1]
                temp[PMAX:APF+1] += self._ppc["gen"][ptr1, PMAX:APF+1]
            else:
                ppc_gen_trim.append(temp)
                temp = self._ppc["gen"][ptr1, :]
                ptr = ptr1
            ptr1 += 1
        ppc_gen_trim.append(temp)
        self._ppc["gen"] = np.asarray(ppc_gen_trim)

    def _merge_branches(self):
        ppc_branch_trim = []
        temp = self._ppc["branch"][0, :]
        ptr = 0
        ptr1 = 1
        while ptr1 < self._ppc["branch"].shape[0]:
            if np.all(self._ppc["branch"][ptr, F_BUS:T_BUS+1] == self._ppc["branch"][ptr1, GEN_BUS:T_BUS+1]):
                temp[BR_R: RATE_C+1] += self._ppc["branch"][ptr1, BR_R: RATE_C+1]
            else:
                ppc_branch_trim.append(temp)
                temp = self._ppc["branch"][ptr1, :]
                ptr = ptr1
            ptr1 += 1
        ppc_branch_trim.append(temp)
        self._ppc["branch"] = np.asarray(ppc_branch_trim)

    def get_ppc(self):
        return self.ppc

    def print_ppc(self):
        print (self.ppc)

    def get_load_demand(self):
        load_demand = self._ppc["bus"][:, PD] / (self._ppc["baseMVA"] * self._power_generation_preprocess_scale)
        return load_demand