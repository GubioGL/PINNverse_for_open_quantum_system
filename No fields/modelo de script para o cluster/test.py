import numpy    as np
import torch    as tc
import qutip    as qt
import pandas   as pd
import torch.nn as nn
import random
import os

from tqdm import tqdm
########################################### FUNCTION for run de algorithm ##################################################  

def data_qubit_two_crosstalk(lista_J,dissipation,tfinal,N,O_op,device="cpu"):
    # Operadores de Pauli para cada qubit
    XX,XY,XZ = qt.tensor(qt.sigmax(),qt.sigmax()),qt.tensor(qt.sigmax(),qt.sigmay()),qt.tensor(qt.sigmax(),qt.sigmaz())
    YX,YY,YZ = qt.tensor(qt.sigmay(),qt.sigmax()),qt.tensor(qt.sigmay(),qt.sigmay()),qt.tensor(qt.sigmay(),qt.sigmaz())
    ZX,ZY,ZZ = qt.tensor(qt.sigmaz(),qt.sigmax()),qt.tensor(qt.sigmaz(),qt.sigmay()),qt.tensor(qt.sigmaz(),qt.sigmaz())
    operadores = [XX,XY,XZ,
                  YX,YY,YZ,
                  ZX,ZY,ZZ]
    
    H = 0
    for i in range(len(lista_J)):
        H += 0.5*lista_J[i]*operadores[i]

    # Hamiltonian Lindbladian
    c_ops = [np.sqrt(dissipation[0])*qt.tensor(qt.sigmam(), qt.qeye(2)),
             np.sqrt(dissipation[1])*qt.tensor(qt.sigmaz(), qt.qeye(2)),
             np.sqrt(dissipation[2])*qt.tensor(qt.qeye(2) , qt.sigmam()),
             np.sqrt(dissipation[3])*qt.tensor(qt.qeye(2) , qt.sigmaz()),]
    # Estado inicial (cada qubit na superposição de |0> e |1>)
    # |+> = (|0> + |1>)/sqrt(2)
    theta1  = np.pi/4
    phi1    = 0.0 #np.pi/3
    ket_plus1 = (np.cos(theta1)*qt.basis(2, 0)+np.sin(theta1)*np.exp(1j*phi1)*qt.basis(2, 1))
    theta2  = np.pi/4
    phi2    = 0.0 #np.pi/5
    ket_plus2 = (np.cos(theta2)*qt.basis(2, 0)+np.sin(theta2)*np.exp(1j*phi2)*qt.basis(2, 1))

    psi0 = qt.tensor(ket_plus1, ket_plus2)
    
    # Lista de tempos para a evolução
    tlist = np.linspace(0.0, tfinal, N)

    # Solução da equação de Schrödinger
    options = qt.Options(nsteps = 100000, atol = 1e-14, rtol = 1e-14)
    result  = qt.mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=O_op,options=options)

    expect  = tc.tensor( np.array( result.expect),device = device).transpose(0, 1)
    return expect

class Rede(nn.Module):
    def __init__(self, neuronio, activation, input_=1, output_=1, creat_p=False, N_of_paramater=1, dropout_prob=0.0):
        super().__init__()
        self.neuronio = neuronio
        self.output = output_
        self.creat_p = creat_p
        self.N_of_paramater = N_of_paramater
        self.dropout_prob = dropout_prob
        
        # input camada linear
        self.hidden_layers = nn.ModuleList([nn.Linear(input_, neuronio[0])])
        # camadas do meio
        self.hidden_layers.extend([nn.Linear(neuronio[_], neuronio[_+1]) for _ in range(len(self.neuronio)-1)])
        # Última camada linear
        self.output_layer = nn.Linear(neuronio[-1], output_)
        
        # Função de ativação
        self.activation_ = activation
        
        # Camadas de Dropout
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(len(self.neuronio))])
        
        # Criar o parâmetro
        if creat_p:
            self.parametro = nn.Parameter(tc.rand(N_of_paramater))
            
    def forward(self, x):
        for layer, activation, dropout in zip(self.hidden_layers, self.activation_, self.dropouts):
            x = activation(layer(x))
            x = dropout(x)
        x = self.output_layer(x)
        return x

O_op =[ qt.tensor(qt.qeye(2) , qt.sigmax()),
        qt.tensor(qt.qeye(2) , qt.sigmay()),
        qt.tensor(qt.qeye(2) , qt.sigmaz()),
        qt.tensor(qt.sigmax(), qt.qeye(2) ),
        qt.tensor(qt.sigmax(), qt.sigmax()),
        qt.tensor(qt.sigmax(), qt.sigmay()),
        qt.tensor(qt.sigmax(), qt.sigmaz()),
        qt.tensor(qt.sigmay(), qt.qeye(2) ),
        qt.tensor(qt.sigmay(), qt.sigmax()),
        qt.tensor(qt.sigmay(), qt.sigmay()),
        qt.tensor(qt.sigmay(), qt.sigmaz()),
        qt.tensor(qt.sigmaz(), qt.qeye(2) ),
        qt.tensor(qt.sigmaz(), qt.sigmax()),
        qt.tensor(qt.sigmaz(), qt.sigmay()),
        qt.tensor(qt.sigmaz(), qt.sigmaz()),
        ]
    
########################################### files for run de algorithm ##################################################  

# Função que executa o treinamento para um valor específico de seed
def run_parallel(SEED,size_data,std):
    # Definindo a seed
    np.random.seed(SEED)
    tc.manual_seed(SEED) 
    random.seed(SEED)
    ################## Definindo os parametros #######################
    
    Js          = [random.uniform(-1,1) for _ in range(15)]
    dissipation = [random.uniform(0,1) for _ in range(4)]
    tfinal      = 2*np.pi
    N           = 1000
            
    valor_esperado_data = data_qubit_two_crosstalk(Js,dissipation,tfinal,N,O_op,)
    random_data = tc.normal(mean=0.0, std=std, size=valor_esperado_data.shape)
    valor_esperado_data_noisy = valor_esperado_data + random_data   
     
    neuronio   = [50,50]
    X_vector   = Rede(
        neuronio    = neuronio,
        input_      = 1,
        output_     = len(O_op),
        activation  = [tc.nn.Tanh()]*len(neuronio),
        creat_p     = True,
        N_of_paramater= 9+4)
    opt     = tc.optim.Adam(X_vector.parameters(),lr = 0.001 )
    time    =  tc.linspace(
            0,
            tfinal,
            N,
            dtype   = tc.float32,
            requires_grad = True).reshape((-1, 1))
    index_data = np.random.randint(0,N,size=size_data)
    epocas  = 200000
    ########################################################
    ##################### Treino ###########################
    for _ in tqdm(range(epocas)):
        ####### Forward pass #######
        y  = X_vector(time)     
        # ####### Los edo #######
        dX_dt = []
        for i in range(y.shape[1]):
            dX_dt.append(tc.autograd.grad(outputs = y[:, i], 
                                        inputs = time,
                                        grad_outputs = tc.ones_like(y[:, i]),
                                        #retain_graph = True,
                                        create_graph = True)[0])
        dX_dt   = tc.cat(dX_dt, dim=1)
        
        IX,IY,IZ,XI,XX,XY,XZ = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4],y[:, 4:5], y[:, 5:6],y[:, 6:7]
        YI,YX,YY,YZ,ZI,ZX,ZY,ZZ = y[:, 7:8], y[:, 8:9], y[:, 9:10],y[:, 10:11], y[:, 11:12], y[:,12:13], y[:,13:14], y[:,14:15]

        gamma1,gamma2,gamma3,gamma4,JXX,JXY,JXZ,JYX,JYY,JYZ,JZX,JZY,JZZ = X_vector.parametro
        
        LOSS_edo  = 0
        LOSS_edo += (dX_dt[:,0:1] - ( +IX*(-0.5*gamma3 - 2.0*gamma4) +(-1.0*JXZ)*XY +JXY*XZ +(-1.0*JYZ)*YY +(1.0*JYY)*YZ +(-1.0*JZZ)*ZY +(1.0*JZY)*ZZ))**2
        LOSS_edo += (dX_dt[:,1:2] - ( +IY*(-0.5*gamma3 - 2.0*gamma4) +(1.0*JXZ)*XX +(-1.0*JXX)*XZ +JYZ*YX +(-1.0*JYX)*YZ +(1.0*JZZ)*ZX +(-JZX)*ZZ))**2
        LOSS_edo += (dX_dt[:,2:3] - ( (-1.0*gamma3) +IZ*(-1.0*gamma3) +(-1.0*JXY)*XX +(1.0*JXX)*XY +(-1.0*JYY)*YX +JYX*YY +(-1.0*JZY)*ZX +(1.0*JZX)*ZY))**2
        LOSS_edo += (dX_dt[:,3:4] - ( +XI*(-0.5*gamma1 - 2.0*gamma2) +(-1.0*JZX)*YX +(-1.0*JZY)*YY +(-1.0*JZZ)*YZ +JYX*ZX +(1.0*JYY)*ZY +(1.0*JYZ)*ZZ))**2
        LOSS_edo += (dX_dt[:,4:5] - ( +IY*(-1.0*JXZ) +IZ*(1.0*JXY) +XX*(-0.5*gamma1 - 2.0*gamma2 - 0.5*gamma3 - 2.0*gamma4) +(-1.0*JZX)*YI +(1.0*JYX)*ZI))**2
        LOSS_edo += (dX_dt[:,5:6] - ( +IX*JXZ +IZ*(-1.0*JXX) +XY*(-0.5*gamma1 - 2.0*gamma2 - 0.5*gamma3 - 2.0*gamma4) +(-1.0*JZY)*YI +(1.0*JYY)*ZI))**2
        LOSS_edo += (dX_dt[:,6:7] - ( +IX*(-1.0*JXY) +IY*(1.0*JXX) +XI*(-1.0*gamma3) +XZ*(-0.5*gamma1 - 2.0*gamma2 - 1.0*gamma3) +(-1.0*JZZ)*YI +(1.0*JYZ)*ZI))**2
        LOSS_edo += (dX_dt[:,7:8] - ( +(1.0*JZX)*XX +JZY*XY +(1.0*JZZ)*XZ +YI*(-0.5*gamma1 - 2.0*gamma2) +(-1.0*JXX)*ZX +(-1.0*JXY)*ZY +(-JXZ)*ZZ))**2
        LOSS_edo += (dX_dt[:,8:9] - ( +IY*(-1.0*JYZ) +IZ*(1.0*JYY) +JZX*XI +YX*(-0.5*gamma1 - 2.0*gamma2 - 0.5*gamma3 - 2.0*gamma4) +(-1.0*JXX)*ZI))**2
        LOSS_edo += (dX_dt[:,9:10] - ( +IX*(1.0*JYZ) +IZ*(-1.0*JYX) +(1.0*JZY)*XI +YY*(-0.5*gamma1 - 2.0*gamma2 - 0.5*gamma3 - 2.0*gamma4) +(-1.0*JXY)*ZI))**2
        LOSS_edo += (dX_dt[:,10:11] - ( +IX*(-1.0*JYY) +IY*JYX +(1.0*JZZ)*XI +YI*(-1.0*gamma3) +YZ*(-0.5*gamma1 - 2.0*gamma2 - 1.0*gamma3) +(-JXZ)*ZI))**2
        LOSS_edo += (dX_dt[:,11:12] - ( (-1.0*gamma1) +(-1.0*JYX)*XX +(-1.0*JYY)*XY +(-1.0*JYZ)*XZ +(1.0*JXX)*YX +JXY*YY +(1.0*JXZ)*YZ +ZI*(-1.0*gamma1)))**2
        LOSS_edo += (dX_dt[:,12:13] - ( +IX*(-1.0*gamma1) +IY*(-1.0*JZZ) +IZ*(1.0*JZY) +(-1.0*JYX)*XI +(1.0*JXX)*YI +ZX*(-1.0*gamma1 - 0.5*gamma3 - 2.0*gamma4)))**2
        LOSS_edo += (dX_dt[:,13:14] - ( +IX*(1.0*JZZ) +IY*(-1.0*gamma1) +IZ*(-JZX) +(-1.0*JYY)*XI +JXY*YI +ZY*(-1.0*gamma1 - 0.5*gamma3 - 2.0*gamma4)))**2
        LOSS_edo += (dX_dt[:,14:15] - ( +IX*(-1.0*JZY) +IY*(1.0*JZX) +IZ*(-1.0*gamma1) +(-1.0*JYZ)*XI +(1.0*JXZ)*YI +ZI*(-1.0*gamma3) +ZZ*(-1.0*gamma1 - 1.0*gamma3)))**2
        LOSS_edo = LOSS_edo.mean() 
        ####### loss data(expected values) #######
        LOSS_data = tc.mean((y[index_data,:]  - valor_esperado_data_noisy[index_data,:])**2) 
        
        ####### Loss total #######
        loss_i = LOSS_edo*0.01 + LOSS_data
        
        ####### Backpropagation #######
        opt.zero_grad()
        loss_i.backward()
        opt.step()  
    ##################### FIM do treino #####################
    parametro_treino = { # ESSA PARTE DE SALVAR OS PARAMETRO DE TREINO TA ERRADA !!!!!!!!!!!!!!
        'gamma1': [dissipation[0]],
        'gamma2': [dissipation[1]],
        'gamma3': [dissipation[2]],
        'gamma4': [dissipation[3]],
        'JXX': [Js[0]],
        'JXY': [Js[1]],
        'JXZ': [Js[2]],
        'JYX': [Js[3]],
        'JYY': [Js[4]],
        'JYZ': [Js[5]],
        'JZX': [Js[6]],
        'JZY': [Js[7]],
        'JZZ': [Js[8]]}

    parametro_previsto = {
        'gamma1': [X_vector.parametro[0].item()],
        'gamma2': [X_vector.parametro[1].item()],
        'gamma3': [X_vector.parametro[2].item()],
        'gamma4': [X_vector.parametro[3].item()],
        'JXX': [X_vector.parametro[4].item()],
        'JXY': [X_vector.parametro[5].item()],
        'JXZ': [X_vector.parametro[6].item()],
        'JYX': [X_vector.parametro[7].item()],
        'JYY': [X_vector.parametro[8].item()],
        'JYZ': [X_vector.parametro[9].item()],
        'JZX': [X_vector.parametro[10].item()],
        'JZY': [X_vector.parametro[11].item()],
        'JZZ': [X_vector.parametro[12].item()]
        }

    data = {'treino': parametro_treino, 'previsto': parametro_previsto}
    df = pd.DataFrame(data)
    df.to_csv(f"parametro_nofields_N{size_data}_seed{SEED}_std{std}.csv")
    print(f"Finalizado N{size_data}_seed{SEED}_std{std} \n")


if __name__ == "__main__":
    # Obter o ID da tarefa a partir da variável de ambiente
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
    # for size_data_index in [5,10,15,20,25]:
    #     run_parallel(task_id,size_data_index,std=0)      

    size_data_index = 50
    for std in [0,0.02,0.04,0.06,0.08,0.1]:
        run_parallel(task_id,size_data_index,std=std) 