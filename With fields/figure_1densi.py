# Gráfico do erro MAPE em função de N_c para cada parâmetro individualmente e para a média dos Js (com barras de erro mínimo e máximo)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do range de N_c
Nc_values       = [50]  # Apenas N=50
parametros_gamma= ['gamma1', 'gamma2', 'gamma3', 'gamma4']
parametros_J    = ['JXX', 'JXY', 'JXZ', 'JYX', 'JYY', 'JYZ', 'JZX', 'JZY', 'JZZ', 'JIX', 'JIY', 'JIZ', 'JXI', 'JYI', 'JZI']


# Armazenar os resultados agregados
erro_gamma_Nc       = []
erro_js_Nc          = []
erro_gamma_Nc_min   = []
erro_js_Nc_min      = []
erro_gamma_Nc_max   = []
erro_js_Nc_max      = []

# Armazenar todos os pontos individuais para cada gamma
all_gamma_points = {g: [] for g in parametros_gamma}


for N in Nc_values:
    erro_gamma  = []
    erro_js     = []
    for i in range(5, 101):
        caminho     = f"C:/Users/Gubio/CODESACE/Pinn inverse for opem quantum system/With fields/data/parametro_withfields_N{N}_seed{i}_std0.csv"
        df          = pd.read_csv(caminho, index_col=0)
        valor_real  = df['treino'].str.strip('[]').astype(float) 
        valor_previsto = df['previsto'].str.strip('[]').astype(float) 
        erro_abs    = np.abs(valor_real - valor_previsto)/abs(valor_real)  # Erro relativo absoluto
        # Salva todos os pontos individuais para cada gamma
        for idx, g in enumerate(parametros_gamma):
            all_gamma_points[g].append(erro_abs[g])
        erro_gamma.append(erro_abs[parametros_gamma].to_numpy())
        erro_js.append(np.mean(erro_abs[parametros_J].to_numpy()))
    # Média, mínimo e máximo dos erros para cada parâmetro
    erro_gamma_Nc.append(np.mean(erro_gamma,0))
    erro_js_Nc.append(np.mean(erro_js))
    erro_gamma_Nc_min.append(np.min(erro_gamma,0))
    erro_js_Nc_min.append(np.min(erro_js))
    erro_gamma_Nc_max.append(np.max(erro_gamma,0))
    erro_js_Nc_max.append(np.max(erro_js))


# Plot
plt.figure(figsize=(8,6))


# Novo eixo x categórico para cada parâmetro e J_medio
param_labels = [fr'$\gamma_{{{i+1}}}$' for i in range(len(parametros_gamma))] + [r'$J_{médio}$']
x_pos = np.arange(len(parametros_gamma) + 1)
colors = ['r','g','b','m','k']

# Plotar todos os pontos individuais para cada parâmetro
for idx, g in enumerate(parametros_gamma):
    y = all_gamma_points[g]
    x = np.full_like(y, x_pos[idx], dtype=float)
    plt.plot(x, y, 'o', color=colors[idx], alpha=0.5)

# Adicionar todos os pontos individuais para J_medio
all_js_points = erro_js  # erro_js contém todos os valores individuais para N=50
x_js = np.full_like(all_js_points, x_pos[-1], dtype=float)
plt.plot(x_js, all_js_points, 'o', color=colors[-1], alpha=0.5)

# Plotar média e barras de erro para cada parâmetro
erro_gamma_Nc = np.array(erro_gamma_Nc)
erro_gamma_Nc_min = np.array(erro_gamma_Nc_min)
erro_gamma_Nc_max = np.array(erro_gamma_Nc_max)
means = list(erro_gamma_Nc[0]) + [erro_js_Nc[0]]
yerr_lower = list(erro_gamma_Nc[0] - erro_gamma_Nc_min[0]) + [erro_js_Nc[0] - erro_js_Nc_min[0]]
yerr_upper = list(erro_gamma_Nc_max[0] - erro_gamma_Nc[0]) + [erro_js_Nc_max[0] - erro_js_Nc[0]]
yerr = [yerr_lower, yerr_upper]
plt.errorbar(x_pos, means, yerr=yerr, fmt='D', color='k', capsize=8, linewidth=2.2, label='Média ± min/max')

plt.yscale('log')
plt.ylabel('MSE', fontsize=22)
plt.xlabel('Parâmetro', fontsize=22)
plt.yticks(fontsize=22)
# plt.ylim(5e-15, 10)
plt.xticks(x_pos, param_labels, fontsize=22)
plt.tick_params(axis='both', which='both', direction='in', length=4, width=1.5)
plt.legend(loc='upper right', fontsize=20, framealpha=0.8, handletextpad=0.01, columnspacing=0.1)
plt.tight_layout()
#plt.savefig("MSE_vs_Ndatas_N50_gammas_separados.png", dpi=300)
plt.show()