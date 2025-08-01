# ингибирующие слой 2/3
L23_I_Axax23 = split_population([HHNeuron(inh=True) for _ in range(N_L23_I)], 3)
L23_I_Bask23 = split_population([HHNeuron(inh=True) for _ in range(N_L23_I)], 3)
L23_I_LTS23 = split_population([HHNeuron(inh=True) for _ in range(N_L23_I)], 3)

# возбуждающие слой 2/3
L23_E_SyppyrRS = split_population([HHNeuron() for _ in range(N_L23_E)], 2)
L23_E_SyppyrFRB = split_population([HHNeuron() for _ in range(N_L23_E)], 2)

# возбуждающие и ингибирующие слой 4
L4_E_Spinstel4 = split_population([HHNeuron() for _ in range(N_L4_E)], 1)
L4_I_LTS4 = split_population([HHNeuron(inh=True) for _ in range(N_L4_I)], 1)

# возбуждающие слой 5
L5_E_TuftRS5 = split_population([HHNeuron() for _ in range(N_L5_E)], 2)
L5_E_TuftIB5 = split_population([HHNeuron() for _ in range(N_L5_E)], 2)

# ингбирующие слой 5 и 6
L56_I_Bask56 = split_population([HHNeuron(inh=True) for _ in range(N_L5_I+N_L6_I)], 3)
L56_I_Axax56 = split_population([HHNeuron(inh=True) for _ in range(N_L5_I+N_L6_I)], 3)
L56_I_LTS56 = split_population([HHNeuron(inh=True) for _ in range(N_L5_I+N_L6_I)], 3)

# возбуждающие слой 6
L6_E_NontuftRS6 = split_population([HHNeuron() for _ in range(N_L6_E)], 1)

#synapses_TCR_to_nRT, netcons_TCR_to_nRT = connect_exc(thalamus_E_TCR, thalamus_I_nRT)
#synapses_TCR_to_L4, netcons_TCR_to_L4 = connect_exc(thalamus_E_TCR, L4_E_Spinstel4)

synapses_TCR_to_nRT, netcons_TCR_to_nRT = connect_exc(thalamus_E_TCR, thalamus_I_nRT)
synapses_TCR_to_L4, netcons_TCR_to_L4 = connect_exc_gauss(thalamus_E_TCR, L4_E_Spinstel4)

synapses_L4_to_L4, netcons_L4_to_L4 = connect_exc_gauss(L4_E_Spinstel4, L4_E_Spinstel4)
synapses_L4_to_L4_I_LTS4, netcons_L4_to_L4_I_LTS4 = connect_exc(L4_E_Spinstel4, L4_I_LTS4)
synapses_L4_to_L23_RS, netcons_L4_to_L23_RS = connect_exc(L4_E_Spinstel4, L23_E_SyppyrRS)
synapses_L4_to_L23_FRB, netcons_L4_to_L23_FRB = connect_exc(L4_E_Spinstel4, L23_E_SyppyrFRB)
synapses_L4_to_L23_LTS, netcons_L4_to_L23_LTS = connect_exc(L4_E_Spinstel4, L23_I_LTS23)
synapses_L4_to_L5_TuftRS, netcons_L4_to_L5_TuftRS = connect_exc(L4_E_Spinstel4, L5_E_TuftRS5)
synapses_L4_to_L5_TuftIB, netcons_L4_to_L5_TuftIB = connect_exc(L4_E_Spinstel4, L5_E_TuftIB5)

synapses_L23_RS_to_RS, netcons_L23_RS_to_RS = connect_exc(L23_E_SyppyrRS, L23_E_SyppyrRS)
synapses_L23_FRB_to_FRB, netcons_L23_FRB_to_FRB = connect_exc(L23_E_SyppyrFRB, L23_E_SyppyrFRB)
synapses_L23_RS_to_FRB, netcons_L23_RS_to_FRB = connect_exc(L23_E_SyppyrRS, L23_E_SyppyrFRB)
synapses_L23_FRB_to_RS, netcons_L23_FRB_to_RS = connect_exc(L23_E_SyppyrFRB, L23_E_SyppyrRS)

synapses_L23_RS_to_TuftRS5, netcons_L23_RS_to_TuftRS5 = connect_exc(L23_E_SyppyrRS, L5_E_TuftRS5)
synapses_L23_RS_to_TuftIB5, netcons_L23_RS_to_TuftIB5 = connect_exc(L23_E_SyppyrRS, L5_E_TuftIB5)
synapses_L23_RS_to_L6, netcons_L23_RS_to_L6 = connect_exc(L23_E_SyppyrRS, L6_E_NontuftRS6)
synapses_L23_FRB_to_TuftRS5, netcons_L23_FRB_to_TuftRS5 = connect_exc(L23_E_SyppyrFRB, L5_E_TuftRS5)
synapses_L23_FRB_to_TuftIB5, netcons_L23_FRB_to_TuftIB5 = connect_exc(L23_E_SyppyrFRB, L5_E_TuftIB5)
synapses_L23_FRB_to_L6, netcons_L23_FRB_to_L6 = connect_exc(L23_E_SyppyrFRB, L6_E_NontuftRS6)

synapses_L23_RS_to_LTS23, netcons_L23_RS_to_LTS23 = connect_exc(L23_E_SyppyrRS, L23_I_LTS23)
synapses_L23_RS_to_Bask23, netcons_L23_RS_to_Bask23 = connect_exc(L23_E_SyppyrRS, L23_I_Bask23)
synapses_L23_RS_to_LTS56, netcons_L23_RS_to_LTS56 = connect_exc(L23_E_SyppyrRS, L56_I_LTS56)
synapses_L23_RS_to_L4, netcons_L23_RS_to_L4 = connect_exc(L23_E_SyppyrRS, L4_E_Spinstel4)

synapses_L23_FRB_to_LTS23, netcons_L23_FRB_to_LTS23 = connect_exc(L23_E_SyppyrFRB, L23_I_LTS23)
synapses_L23_FRB_to_Bask23, netcons_L23_FRB_to_Bask23 = connect_exc(L23_E_SyppyrFRB, L23_I_Bask23)
synapses_L23_FRB_to_Axax23, netcons_L23_FRB_to_Axax23 = connect_exc(L23_E_SyppyrFRB, L23_I_Axax23)
synapses_L23_FRB_to_LTS56, netcons_L23_FRB_to_LTS56 = connect_exc(L23_E_SyppyrFRB, L56_I_LTS56)
synapses_L23_FRB_to_Bask56, netcons_L23_FRB_to_Bask56 = connect_exc(L23_E_SyppyrFRB, L56_I_Bask56)
synapses_L23_FRB_to_Axax56, netcons_L23_FRB_to_Axax56 = connect_exc(L23_E_SyppyrFRB, L56_I_Axax56)

synapses_TuftRS5_to_TuftRS5, netcons_TuftRS5_to_TuftRS5 = connect_exc(L5_E_TuftRS5, L5_E_TuftRS5)
synapses_TuftRS5_to_TuftIB5, netcons_TuftRS5_to_TuftIB5 = connect_exc(L5_E_TuftRS5, L5_E_TuftIB5)
synapses_TuftRS5_to_L6, netcons_TuftRS5_to_L6 = connect_exc(L5_E_TuftRS5, L6_E_NontuftRS6)
synapses_TuftRS5_to_LTS56, netcons_TuftRS5_to_LTS56 = connect_exc(L5_E_TuftRS5, L56_I_LTS56)
synapses_TuftRS5_to_Bask56, netcons_TuftRS5_to_Bask56 = connect_exc(L5_E_TuftRS5, L56_I_Bask56)

synapses_TuftIB5_to_TuftIB5, netcons_TuftIB5_to_TuftIB5 = connect_exc(L5_E_TuftIB5, L5_E_TuftIB5)
synapses_TuftIB5_to_TuftRS5, netcons_TuftIB5_to_TuftRS5 = connect_exc(L5_E_TuftIB5, L5_E_TuftRS5)
synapses_TuftIB5_to_L6, netcons_TuftIB5_to_L6 = connect_exc(L5_E_TuftIB5, L6_E_NontuftRS6)
synapses_TuftIB5_to_LTS56, netcons_TuftIB5_to_LTS56 = connect_exc(L5_E_TuftIB5, L56_I_LTS56)

synapses_L6_to_TuftRS5, netcons_L6_to_TuftRS5 = connect_exc(L6_E_NontuftRS6, L5_E_TuftRS5)
synapses_L6_to_TuftIB5, netcons_L6_to_TuftIB5 = connect_exc(L6_E_NontuftRS6, L5_E_TuftIB5)
#synapses_L6_to_TCR, netcons_L6_to_TCR = connect_exc(L6_E_NontuftRS6, thalamus_E_TCR)  # закомментировано

# L2/3: LTS и Basket
synapses_LTS23_to_LTS23, netcons_LTS23_to_LTS23 = connect_inh(L23_I_LTS23, L23_I_LTS23)
synapses_LTS23_to_RS, netcons_LTS23_to_RS = connect_inh(L23_I_LTS23, L23_E_SyppyrRS)
synapses_LTS23_to_FRB, netcons_LTS23_to_FRB = connect_inh(L23_I_LTS23, L23_E_SyppyrFRB)
synapses_LTS23_to_Bask23, netcons_LTS23_to_Bask23 = connect_inh(L23_I_LTS23, L23_I_Bask23)
synapses_LTS23_to_L5_TuftRS, netcons_LTS23_to_L5_TuftRS = connect_inh(L23_I_LTS23, L5_E_TuftRS5)
synapses_LTS23_to_L5_TuftIB, netcons_LTS23_to_L5_TuftIB = connect_inh(L23_I_LTS23, L5_E_TuftIB5)
synapses_LTS23_to_L6, netcons_LTS23_to_L6 = connect_inh(L23_I_LTS23, L6_E_NontuftRS6)

synapses_Bask23_to_LTS23, netcons_Bask23_to_LTS23 = connect_inh(L23_I_Bask23, L23_I_LTS23)
synapses_Bask23_to_RS, netcons_Bask23_to_RS = connect_inh(L23_I_Bask23, L23_E_SyppyrRS)
synapses_Bask23_to_FRB, netcons_Bask23_to_FRB = connect_inh(L23_I_Bask23, L23_E_SyppyrFRB)
synapses_Bask23_to_Bask23, netcons_Bask23_to_Bask23 = connect_inh(L23_I_Bask23, L23_I_Bask23)
synapses_Bask23_to_L5_TuftRS, netcons_Bask23_to_L5_TuftRS = connect_inh(L23_I_Bask23, L5_E_TuftRS5)
synapses_Bask23_to_L5_TuftIB, netcons_Bask23_to_L5_TuftIB = connect_inh(L23_I_Bask23, L5_E_TuftIB5)
synapses_Bask23_to_L6, netcons_Bask23_to_L6 = connect_inh(L23_I_Bask23, L6_E_NontuftRS6)

# L4: LTS
synapses_LTS4_to_L4, netcons_LTS4_to_L4 = connect_inh(L4_I_LTS4, L4_E_Spinstel4)

# L5/6: ингибиторы
synapses_LTS56_to_LTS56, netcons_LTS56_to_LTS56 = connect_inh(L56_I_LTS56, L56_I_LTS56)
synapses_LTS56_to_LTS23, netcons_LTS56_to_LTS23 = connect_inh(L56_I_LTS56, L23_I_LTS23)
synapses_LTS56_to_RS, netcons_LTS56_to_RS = connect_inh(L56_I_LTS56, L23_E_SyppyrRS)
synapses_LTS56_to_FRB, netcons_LTS56_to_FRB = connect_inh(L56_I_LTS56, L23_E_SyppyrFRB)
synapses_LTS56_to_L5_TuftRS, netcons_LTS56_to_L5_TuftRS = connect_inh(L56_I_LTS56, L5_E_TuftRS5)
synapses_LTS56_to_L5_TuftIB, netcons_LTS56_to_L5_TuftIB = connect_inh(L56_I_LTS56, L5_E_TuftIB5)
synapses_LTS56_to_L6, netcons_LTS56_to_L6 = connect_inh(L56_I_LTS56, L6_E_NontuftRS6)

synapses_Bask56_to_Bask56, netcons_Bask56_to_Bask56 = connect_inh(L56_I_Bask56, L56_I_Bask56)
synapses_Bask56_to_L5_TuftRS, netcons_Bask56_to_L5_TuftRS = connect_inh(L56_I_Bask56, L5_E_TuftRS5)
synapses_Bask56_to_L5_TuftIB, netcons_Bask56_to_L5_TuftIB = connect_inh(L56_I_Bask56, L5_E_TuftIB5)
synapses_Bask56_to_L6, netcons_Bask56_to_L6 = connect_inh(L56_I_Bask56, L6_E_NontuftRS6)

synapses_Axax56_to_TuftRS5, netcons_Axax56_to_TuftRS5 = connect_inh(L56_I_Axax56, L5_E_TuftRS5)

# Таламус: ингибиторы
synapses_nRT_to_nRT, netcons_nRT_to_nRT = connect_inh(thalamus_I_nRT, thalamus_I_nRT)
# synapses_nRT_to_TCR, netcons_nRT_to_TCR = connect_inh(thalamus_I_nRT, thalamus_E_TCR)  # закомментировано

h.tstop = 200
h.finitialize(-65)
h.continuerun(h.tstop)
