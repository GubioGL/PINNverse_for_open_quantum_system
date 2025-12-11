# Gráfico do erro MAPE em função de N_c para cada parâmetro individualmente e para a média dos Js (com barras de erro mínimo e máximo)
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

# Parâmetros do range de N_c
Nc_values       = [0,0.02,0.04,0.06,0.08,0.1]
parametros_gamma= ['gamma1', 'gamma2', 'gamma3', 'gamma4']
parametros_J    = ['JXX', 'JXY', 'JXZ', 'JYX', 'JYY', 'JYZ', 'JZX', 'JZY', 'JZZ', 'JIX', 'JIY', 'JIZ', 'JXI', 'JYI', 'JZI']

#Armazenar os resultados
erro_gamma_Nc       = []
erro_js_Nc          = []

erro_gamma_Nc_min   = []
erro_js_Nc_min      = []

erro_gamma_Nc_max   = []
erro_js_Nc_max      = []

for N in Nc_values:
    erro_gamma  = []
    erro_js     = []
    for i in range(5, 101):
        
        caminho     = f"C:/Users/Gubio/CODESACE/Pinn inverse for opem quantum system/With fields/data/parametro_withfields_N50_seed{i}_std{N}.csv"
        try:
            
            df          = pd.read_csv(caminho, index_col=0)
            valor_real  = df['treino'].str.strip('[]').astype(float) 
            valor_previsto = df['previsto'].str.strip('[]').astype(float) 
            
            erro_abs    = (valor_real - valor_previsto)**2
            
            erro_gamma.append(erro_abs[parametros_gamma].to_numpy())
            erro_js.append(np.mean(erro_abs[parametros_J].to_numpy()))

        except FileNotFoundError:
            print(f"parametro_withfields_N{N}_seed{i}_std0.csv")
            continue

    # Média, mínimo e máximo dos erros para cada parâmetro
    erro_gamma_Nc.append(np.mean(erro_gamma))
    erro_js_Nc.append(np.mean(erro_js))

    erro_gamma_Nc_min.append(np.min(erro_gamma))
    erro_js_Nc_min.append(np.min(erro_js))
    erro_gamma_Nc_max.append(np.max(erro_gamma))
    erro_js_Nc_max.append(np.max(erro_js))

# Plot
plt.rcParams.update({'font.size': 24})
plt.figure(figsize=(8,6))
Nc_arr = np.array(Nc_values)

# Plot Js médio com traço mais espesso
yerr_J = [np.array(erro_js_Nc) - np.array(erro_js_Nc_min), np.array(erro_js_Nc_max) - np.array(erro_js_Nc)]
plt.errorbar(Nc_arr, erro_js_Nc, yerr=yerr_J, fmt='ks', capsize=8, label=r'$J_{médio}$', linewidth=2.2)

erro_gamma_Nc       = np.array(erro_gamma_Nc)
erro_gamma_Nc_min   = np.array(erro_gamma_Nc_min)
erro_gamma_Nc_max   = np.array(erro_gamma_Nc_max)

yerr_g = [erro_gamma_Nc - erro_gamma_Nc_min , erro_gamma_Nc_max - erro_gamma_Nc]
plt.errorbar(Nc_arr, erro_gamma_Nc, yerr=yerr_g, fmt='o', color='r', capsize=8, linewidth=2.2,label=r'$\gamma_{médio}$') 


plt.yscale('log')
plt.ylabel('MSE')
plt.xlabel('Desvio padrão')   

plt.xticks(Nc_values)
plt.tick_params(axis='both', which='both', direction='in', length=4, width=1.5)
plt.legend(loc='lower right', ncol=2, fontsize=20, framealpha=0.8, handletextpad=0.01, columnspacing=0.1)

plt.grid(axis='y', linestyle='--', linewidth=0.5)
# plt.gca().set_ylim(1e-6, 1e0)  # Set y-axis limits
plt.gca().yaxis.set_major_locator(plt.FixedLocator([1e0, 1e-2, 1e-4, 1e-6,1e-9,1e-12]))
plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))


plt.tight_layout()
plt.savefig("MSE_vs_ruido.pdf", dpi=500 )
plt.show()
print(erro_gamma_Nc)
print(erro_js_Nc)