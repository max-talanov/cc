import matplotlib.pyplot as plt

# Только возбуждающие группы
'''
excitatory_groups = {
    "TCR (таламус, возбужд.)": thalamus_E_TCR,
    "L2/3 SyppyrRS": flatten(L23_E_SyppyrRS),
    "L2/3 SyppyrFRB": flatten(L23_E_SyppyrFRB),
    "L4 Spinstel": flatten(L4_E_Spinstel4),
    "L5 TuftRS": flatten(L5_E_TuftRS5),
    "L5 TuftIB": flatten(L5_E_TuftIB5),
    "L6 NontuftRS": flatten(L6_E_NontuftRS6)
}
'''
excitatory_groups = {
    "TCR (таламус, возбужд.)": thalamus_E_TCR,
    "L4 Spinstel": flatten(L4_E_Spinstel4)
}

plt.figure(figsize=(12, 6))

for name, neurons in excitatory_groups.items():
    neuron = neurons[0]
    v = neuron.vvec.as_numpy()
    t = neuron.tvec.as_numpy()
    mask = t <= 100
    plt.plot(t[mask], v[mask], label=name)



plt.title("Мембранный потенциал (возбуждающие нейроны из разных слоёв)")
plt.xlabel("Время (мс)")
plt.ylabel("Потенциал (мВ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Ингибирующие группы
inhibitory_groups = {
    "thalamus_I_nRT": thalamus_I_nRT,
    "L2/3 Basket": flatten(L23_I_Bask23),
    "L2/3 LTS": flatten(L23_I_LTS23),
    "L2/3 AxoAxonic": flatten(L23_I_Axax23),
    "L4 LTS": flatten(L4_I_LTS4),
    "L5/6 Basket": flatten(L56_I_Bask56),
    "L5/6 LTS": flatten(L56_I_LTS56),
    "L5/6 AxoAxonic": flatten(L56_I_Axax56),
}

plt.figure(figsize=(12, 6))

for name, neurons in inhibitory_groups.items():
    neuron = neurons[0]  # Выбираем первого нейрона в группе
    v = neuron.vvec.as_numpy()
    t = neuron.tvec.as_numpy()
    mask = t <= 150
    plt.plot(t[mask], v[mask], label=name)

plt.title("Мембранный потенциал (ингибирующие нейроны из разных слоёв)")
plt.xlabel("Время (мс)")
plt.ylabel("Потенциал (мВ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 6))
#neurons = flatten(L4_E_Spinstel4)[:10]
neurons = flatten(L6_E_NontuftRS6)[:]

for i, neuron in enumerate(neurons):
    v = np.array(neuron.vvec)
    t = np.array(neuron.tvec)
    mask = t <= 100  # ограничим по времени

    plt.plot(t[mask], v[mask], label=f"Neuron {i}", alpha=0.6)

plt.title("Мембранные потенциалы нейронов слоя L4")
plt.xlabel("Время (мс)")
plt.ylabel("Потенциал (мВ)")
plt.grid(True)
# plt.legend()  # можно включить, но будет много подписей
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

t_start = 0     # начальное время
t_end = 200    # конечное время

import numpy as np

def extract_spike_times(neuron, threshold=0, refractory_period=2.0):
    v = np.array(neuron.vvec)
    t = np.array(neuron.tvec)

    spike_times = []
    last_spike_time = -np.inf

    for i in range(1, len(v)):
        if v[i-1] < threshold and v[i] >= threshold:
            if (t[i] - last_spike_time) >= refractory_period:
                spike_times.append(t[i])
                last_spike_time = t[i]

    return np.array(spike_times)

neurons = [flatten(L4_E_Spinstel4)[i] for i in range(20)]

for i, neuron in enumerate(neurons):
    spike_times = extract_spike_times(neuron, threshold=10, refractory_period=3.0)
    print(f"Нейрон {i}: {len(spike_times)} спайков → {np.round(spike_times).astype(int)} мс")

# Параметры временной оси
t_start = 0
t_end = 200
bin_size = 1  # ширина окна (мс)
bins = np.arange(t_start, t_end + bin_size, bin_size)

# Расплющиваем группу, если она разбита на подгруппы
L4_group = flatten(L4_E_Spinstel4)
#L4_group = flatten(L23_E_SyppyrRS)

spike_activity = np.zeros(len(bins) - 1)

for neuron in L4_group:
    spike_times = extract_spike_times(neuron, threshold=0, refractory_period=2.0)
    spike_bins = np.digitize(spike_times, bins) - 1
    unique_bins = np.unique(spike_bins[(spike_bins >= 0) & (spike_bins < len(spike_activity))])
    spike_activity[unique_bins] += 1

# --- Построение гистограммы ---
plt.figure(figsize=(12, 5))
plt.bar(bins[:-1], spike_activity, width=bin_size, align='edge', color='skyblue', edgecolor='black')
plt.xlabel("Время (мс)")
plt.ylabel("Число нейронов со спайком")
plt.title("Гистограмма спайковой активности нейронов L4_E_Spinstel4 (extract_spike_times)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

import numpy as np

excitatory_groups = {
    "TCR (таламус, возбужд.)": thalamus_E_TCR,
    "L2/3 SyppyrRS": flatten(L23_E_SyppyrRS),
    "L2/3 SyppyrFRB": flatten(L23_E_SyppyrFRB),
    "L4 Spinstel": flatten(L4_E_Spinstel4),
    "L5 TuftRS": flatten(L5_E_TuftRS5),
    "L5 TuftIB": flatten(L5_E_TuftIB5),
    "L6 NontuftRS": flatten(L6_E_NontuftRS6)
}


def build_spike_histogram(spike_matrix, t_start=0, t_stop=100, bin_size=5):
    bins = np.arange(t_start, t_stop + bin_size, bin_size)
    hist_matrix = []

    for group_name, spike_lists in spike_matrix.items():
        all_spikes = np.concatenate(spike_lists)
        counts, _ = np.histogram(all_spikes, bins=bins)
        hist_matrix.append(counts)

    return np.array(hist_matrix), bins
import matplotlib.pyplot as plt

def plot_spike_heatmap(hist_matrix, bins, group_names):
    plt.figure(figsize=(12, 6))
    im = plt.imshow(hist_matrix, aspect='auto', cmap='viridis', origin='lower')

    plt.colorbar(im, label="Количество спайков")
    plt.xlabel("Время (мс)")
    plt.ylabel("Группы нейронов")
    plt.title("Тепловая карта активности нейронных групп")

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.xticks(ticks=np.arange(len(bin_centers)), labels=np.round(bin_centers).astype(int), rotation=45)
    plt.yticks(ticks=np.arange(len(group_names)), labels=group_names)

    plt.tight_layout()
    plt.show()


# Шаг 1: собрать spike_matrix (как раньше)
spike_matrix = {}
for group_name, neurons in excitatory_groups.items():
    group_spikes = [extract_spike_times(neuron) for neuron in neurons]
    spike_matrix[group_name] = group_spikes

# Шаг 2: построить гистограмму
hist_matrix, bins = build_spike_histogram(spike_matrix, t_start=0, t_stop=h.tstop, bin_size=5)

# Шаг 3: отрисовать тепловую карту
plot_spike_heatmap(hist_matrix, bins, list(spike_matrix.keys()))