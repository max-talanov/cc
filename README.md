# Thalamo-cortical loop
the repository for thalamo-cortical loop modelling
![](CC.png)



# 🧠 Cortical Column Simulation (NEURON + Python)

We model the thalamo-cortical loop using the [NEURON](https://neuron.yale.edu/neuron/) neurosimulator. We model excitatory and inhibitory neurons and visualise there activities.

---

## 📁 Project contents

| File                       | Description                                                                                                                |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `cortical_column.py`       | Main script of the simulation: imports all submodules and runs simulation and does visualisation.                          |
| `create_HHNeuron_group.py` | Defines Neuron class `HHNeuron` (Hodgkin-Huxley) and groups of neurons creation functions.                                 |
| `connect_functions.py`     | Contains functions of neurons connection: `connect_exc` (excitatory projections) and `connect_inh` (inhibitory).           |
| `create_connections.py`    | Setup synaptic connections between all neuronal groups (thalamus, cortical layers).                                        |
| `thalamus_stimulation.py`  | Exeternal stimulation of neurons of thalamus via `NetStim`.                                                                |
| `visualization.py`         | Results visualisation: membraine potentials, spike times, heatmaps etc.                                                    |

---

## 🧠 Biological components

* **TCR (Thalamocortical relay cells)** — thalamus excitatory neurons
* **nRT (nucleus Reticularis Thalami)** — thalamus inhibitory neurons
* **Cortical layers**:

  * L2/3 SyppyrRS
  * L2/3 SyppyrFRB
  * L4 Spinstel
  * L5 TuftRS
  * L5 TuftIB
  * L6 NontuftRS

---

## ⚙️ Dependencies

* Python 3.8+
* [NEURON](https://www.neuron.yale.edu/neuron/download)
* `numpy`
* `matplotlib`

Installation of dependencies:

```bash
pip install numpy matplotlib
```

Установка NEURON см. в [официальной документации](https://neuron.yale.edu/neuron/download).

---

## 🚀 Запуск

Выполните основной скрипт:

```bash
python cortical_column.py
```

Результаты симуляции будут выведены в виде графиков, показывающих изменение потенциалов, время спайков и тепловые карты по группам.

---

## 📊 Возможности

* Моделирование индивидуальных и групповых HH-нейронов
* Возбуждающие и ингибирующие связи
* Внешняя стимуляция `NetStim`
* Запись и анализ спайков
* Гибкая настройка весов, задержек и параметров синапсов
* Визуализация активности по группам

---

## 📎 Пример визуализации

<img width="1189" height="590" alt="potential" src="https://github.com/user-attachments/assets/82561ede-423b-482b-8b3d-584e2185eee0" />


---

## 📌 TODO

* [ ] Добавить STDP/пластичность
* [ ] Визуализация синаптических воздействий во времени
* [ ] Интеграция с 3D-анатомическими данными
* [ ] Сохранение/загрузка конфигурации сети

---




