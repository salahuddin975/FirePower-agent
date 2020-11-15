import numpy as np
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.loadcase import loadcase
from pypower.ext2int import ext2int



class Generators:
    def __init__(self, ppc, ramp_frequency_in_hour = 6):
        self.generators = np.copy(ppc["gen"][:, GEN_BUS].astype("int"))
        self.num_generators = self.generators.size
        self.generators_min_output = np.copy(ppc["gen"][:, PMIN] / ppc["baseMVA"])
        self.generators_max_output = np.copy(ppc["gen"][:, PMAX] / ppc["baseMVA"])
        self.generators_max_ramp = np.copy((ppc["gen"][:, RAMP_10] / ppc["baseMVA"]) * (1 / ramp_frequency_in_hour))

    def get_generators(self):
        return self.generators

    def get_size(self):
        return self.num_generators

    def get_min_outputs(self):
        return self.generators_min_output

    def get_max_outputs(self):
        return self.generators_max_output

    def set_max_outputs(self, max_output):
        self.generators_max_output = np.copy(max_output[self.generators])

    def get_max_ramps(self):
        return  self.generators_max_ramp

    def print_info(self):
        print ("generators: ", self.generators)
        print ("generators min output: ", self.generators_min_output)
        print ("generators max output: ", self.generators_max_output)
        print ("generators max ramp: ", self.generators_max_ramp)


class SimulatorResources():
    def __init__(self, power_file_path, geo_file_path):
        self._ppc = loadcase(power_file_path)
        self._merge_generators()
        self._merge_branches()
        self.ppc = ext2int(self._ppc)

    def _merge_generators(self):
        ppc_gen_trim = []
        temp = self._ppc["gen"][0, :]
        ptr = 0
        ptr1 = 1
        while(ptr1 < self._ppc["gen"].shape[0]):
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
        while(ptr1 < self._ppc["branch"].shape[0]):
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