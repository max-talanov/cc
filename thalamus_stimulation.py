thalamus_E_TCR = [HHNeuron(inh=False) for _ in range(N_thalamus_E)]
thalamus_I_nRT = [HHNeuron(inh=True) for _ in range(N_thalamus_I)]

# Генератор спайков
syn_inputs = []
conns = []
netstims = []


for cell in thalamus_E_TCR:
    netstim = h.NetStim()
    netstim.start = 0      # время первого спайка
    netstim.number = 20      # количество спайков
    netstim.interval = 10   # интервал между спайками
    netstim.noise = 1
    syn = h.ExpSyn(cell.soma(0.5))
    syn.e = 0
    syn.tau = 2.0
    nc = h.NetCon(netstim, syn)
    nc.weight[0] = 0.01
    syn_inputs.append(syn)
    conns.append(nc)
    netstims.append(netstim)