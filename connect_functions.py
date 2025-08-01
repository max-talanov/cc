'''
def connect_exc(source_neurons, target_neurons, e=0, tau=2.0, threshold=0, weight=0.001, delay=3.0):

    source_neurons = flatten(source_neurons)
    target_neurons = flatten(target_neurons)
    netcons = []
    synapses = []

    for src in source_neurons:
        for tgt in target_neurons:
            syn = h.ExpSyn(tgt.soma(0.5))
            syn.e = e
            syn.tau = tau

            nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
            nc.threshold = threshold
            nc.weight[0] = weight
            nc.delay = delay

            synapses.append(syn)
            netcons.append(nc)

    return synapses, netcons
'''
import random

def connect_exc(
    source_neurons,
    target_neurons,
    e=0,
    tau=2.0,
    threshold=0,
    weight_mean=0.001,
    weight_std=0.002,
    delay_mean=3.0,
    delay_std=2
):
    source_neurons = flatten(source_neurons)
    target_neurons = flatten(target_neurons)
    netcons = []
    synapses = []

    for src in source_neurons:
        for tgt in target_neurons:
            syn = h.ExpSyn(tgt.soma(0.5))
            syn.e = e
            syn.tau = tau

            nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
            nc.threshold = threshold

            # Генерация веса и задержки из нормального распределения
            w = max(0.0, random.gauss(weight_mean, weight_std))
            d = max(0.1, random.gauss(delay_mean, delay_std))  # минимальная задержка 0.1 мс

            nc.weight[0] = w
            nc.delay = d

            synapses.append(syn)
            netcons.append(nc)

    return synapses, netcons

def connect_exc_gauss(
    source_neurons,
    target_neurons,
    e=0,
    tau=2.0,
    threshold=0,
    weight_mean=0.001,
    weight_std=0.0009,
    delay_mean=3.0,
    delay_std=2
):
    source_neurons = flatten(source_neurons)
    target_neurons = flatten(target_neurons)
    netcons = []
    synapses = []

    for src in source_neurons:
        for tgt in target_neurons:
            syn = h.ExpSyn(tgt.soma(0.5))
            syn.e = e
            syn.tau = tau

            nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
            nc.threshold = threshold

            # Генерация веса и задержки из нормального распределения
            w = max(0.0, random.gauss(weight_mean, weight_std))
            d = max(0.1, random.gauss(delay_mean, delay_std))  # минимальная задержка 0.1 мс

            nc.weight[0] = w
            nc.delay = d

            synapses.append(syn)
            netcons.append(nc)

    return synapses, netcons


def connect_inh(source_neurons, target_neurons, e=-75, tau=3.0, threshold=0, weight=0.001, delay=2.0):

    source_neurons = flatten(source_neurons)
    target_neurons = flatten(target_neurons)
    netcons = []
    synapses = []

    for src in source_neurons:
        for tgt in target_neurons:
            syn = h.ExpSyn(tgt.soma(0.5))
            syn.e = e
            syn.tau = tau

            nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
            nc.threshold = threshold
            nc.weight[0] = weight
            nc.delay = delay

            synapses.append(syn)
            netcons.append(nc)

    return synapses, netcons