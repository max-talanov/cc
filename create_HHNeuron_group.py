!pip install neuron

from neuron import h, gui
import random

h.load_file('stdrun.hoc')

# Параметры сети
N_thalamus_E = 5
N_thalamus_I = 1
N_L4_E = 24
N_L4_I = 6
N_L23_E = 24
N_L23_I = 6
N_L5_E = 20
N_L5_I = 5
N_L6_E = 16
N_L6_I = 4

h.tstop = 200.0


class HHNeuron:
    def __init__(self, inh=False):
        self.soma = h.Section(name='soma')
        self.soma.L = 20
        self.soma.diam = 20
        self.soma.insert('hh')

        self.inh = inh

        # Запись мембранного потенциала
        self.vvec = h.Vector()
        self.vvec.record(self.soma(0.5)._ref_v)

        self.tvec = h.Vector()
        self.tvec.record(h._ref_t)


# Автоматически расплющиваем вложенные списки
def split_population(population, n_subgroups):
    size = len(population)
    step = size // n_subgroups
    return [population[i*step:(i+1)*step] for i in range(n_subgroups)]

def flatten(population):
    if isinstance(population[0], list):
        return [neuron for subgroup in population for neuron in subgroup]
    return population