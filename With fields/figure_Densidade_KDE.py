# Gráfico do erro MAPE em função de N_c para cada parâmetro individualmente e para a média dos Js (com barras de erro mínimo e máximo)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parâmetros do range de N_c
Nc_values       = [5]  # Apenas N=50 para KDE
parametros_gamma= ['gamma1', 'gamma2', 'gamma3', 'gamma4']
parametros_J    = ['JXX', 'JXY', 'JXZ', 'JYX', 'JYY', 'JYZ', 'JZX', 'JZY', 'JZZ', 'JIX', 'JIY', 'JIZ', 'JXI', 'JYI', 'JZI']


# Armazenar todos os erros individuais para KDE
all_gamma_points = {g: [] for g in parametros_gamma}
all_js_points = []


for N in Nc_values:
    for i in range(5, 101):
        caminho     = f"C:/Users/Gubio/CODESACE/Pinn inverse for opem quantum system/With fields/data/parametro_withfields_N{N}_seed{i}_std0.csv"
        df          = pd.read_csv(caminho, index_col=0)
        valor_real  = df['treino'].str.strip('[]').astype(float) 
        valor_previsto = df['previsto'].str.strip('[]').astype(float) 
        erro_abs    = np.abs(valor_real - valor_previsto)/abs(valor_real)  # Erro relativo absoluto
        for g in parametros_gamma:
            all_gamma_points[g].append(erro_abs[g])
        all_js_points.append(np.mean(erro_abs[parametros_J].to_numpy()))


# KDE Plot
plt.figure(figsize=(8,6))
colors = ['r','g','b','m','k']
labels = [fr'$\gamma_{{{i+1}}}$' for i in range(4)] + [r'$J_{médio}$']

for idx, g in enumerate(parametros_gamma):
    data = np.array(all_gamma_points[g])
    sns.kdeplot(data, bw_adjust=0.5, fill=True, color=colors[idx], label=labels[idx], log_scale=True, alpha=0.5)

# KDE para J_medio
data_j = np.array(all_js_points)
sns.kdeplot(data_j, bw_adjust=0.5, fill=True, color=colors[4], label=labels[4], log_scale=True, alpha=0.5)

plt.xscale('log')
plt.xlabel('MSE', fontsize=22)
plt.ylabel('Densidade KDE', fontsize=22)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
#plt.savefig("KDE_MSE_N50.png", dpi=300)
plt.show()