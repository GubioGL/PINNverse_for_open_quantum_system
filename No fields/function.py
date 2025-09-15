import torch as tc
import numpy as np
import qutip as qt
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Define the loss function
def mse_loss(y_pred, y_true):
    return tc.mean((y_pred - y_true)**2)

# Define the loss function
def msa_loss(y_pred, y_true):
    return tc.mean(abs(y_pred - y_true))

def diagonal(M):
    return  M.diagonal(offset=0,dim1=1, dim2=2)

def expected(A,B):
    return diagonal(A@B)

def commutator(A, B):
    return tc.matmul(A,B) - tc.matmul(B,A)

def Loss_EDO(H_,rho_r,rho_i,O_,tempo,baserho_):
    N_ = tempo.shape[0]
    rho_ = (rho_r + 1j*rho_i).reshape((N_,baserho_,baserho_)) 

    #  Calculando o Comutador do Hamiltoniano com a matriz densidade. 
    H_rho_R = (commutator(H_.real,rho_.imag)+ commutator(H_.imag,rho_.real)).reshape((N_,baserho_**2))
    H_rho_I = (commutator(H_.imag,rho_.imag)- commutator(H_.real,rho_.real)).reshape((N_,baserho_**2))
    
    # Calculanod o termo do Lindbladiano
    # Aumentando a dimensão para fazer a multiplicação
    O_t     = tc.conj(O_.transpose(-2, -1)).unsqueeze(0)# shape [1,len(O_t),baserho_,baserho_)]
    rho_    = rho_.unsqueeze(1) # shape [len(time),1,baserho_,baserho_)]

    Lindb   = O_@rho_@O_t -0.5*(O_@O_t@rho_ + rho_@O_@O_t )
    Lindb   = Lindb.sum(dim=1)
    Lindb_R = (Lindb.real).reshape((N_,baserho_**2))
    Lindb_I = (Lindb.imag).reshape((N_,baserho_**2))
    
    # Calculando o gradiente de drho_dt separando a parte real e imagina
    loss_edo = 0
    for i in range(baserho_**2):
        drho_dt_real = tc.autograd.grad(outputs = rho_r[:,i], 
                            inputs = tempo,
                            grad_outputs = tc.ones_like(rho_r[:,i]),
                            retain_graph = True,
                            create_graph = True
                            )[0][:,0]

        drho_dt_imag = tc.autograd.grad(outputs = rho_i[:,i], 
                            inputs = tempo,
                            grad_outputs = tc.ones_like(rho_i[:,i]),
                            retain_graph = True,
                            create_graph = True
                            )[0][:,0]
        
        # Von Neuman equation
        lambdas = tc.tensor([2,1])
        loss_edo += tc.mean( 
            lambdas[0]*(drho_dt_real - H_rho_R[:,i]- Lindb_R[:,i])**2 + 
            lambdas[1]*(drho_dt_imag - H_rho_I[:,i]- Lindb_I[:,i])**2 )
    return loss_edo 

def expected_plot( rho_,O_,expected_data,time_,save_plot=None):
    if expected_data.shape == 1:
        fig, ax = plt.subplots()
        v_esperados = expected(rho_, O_).sum(dim=-1).real

        plt.plot(time_.detach().numpy(), v_esperados.detach().numpy(), "r.", label="Neural Network")
        plt.plot(time_.detach().numpy(), expected_data.detach().numpy(), "k.", label="Data")
        plt.xlabel("Time")
        plt.legend()
        #plt.ylim([-1.1,1.1])
        if save_plot == True:
            fig.savefig("fig.png",dpi =500)
        plt.show()
    else:
        fig, ax = plt.subplots()
        for i in range(len(O_)):
            v_esperados = expected(rho_, O_[i]).sum(dim=-1).real
            plt.plot(time_.detach().numpy(), v_esperados.detach().numpy(), "r.", label="Neural Network")
            plt.plot(time_.detach().numpy(), expected_data[:,i].detach().numpy(), "k.", label="Data")
        plt.xlabel("Time")
        #plt.ylabel("<sigma_z>")
        plt.legend()
        #plt.ylim([-1.1,1.1])
        if save_plot == True:
            plt.savefig("fig.png",dpi=500)
        plt.show()

def plots_rho(rho_NNR=0,rho_NNI=0,rho_data=0 ):
    fig, axs = plt.subplots(nrows=3, ncols=2 , figsize=(12,4), sharex=True)

    im =axs[0,0].imshow(rho_NNR.detach().numpy().T,cmap="jet")
    axs[0,0].set_title(r"$\mathcal{R}(\rho_{NN})$")
    axs[0,0].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[1,0].imshow(rho_data.real.T.detach().numpy(),cmap="jet")
    axs[1,0].set_title(r"$\mathcal{R}(\hat{\rho}_t)$")
    axs[1,0].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[2,0].imshow(abs(rho_data.real-rho_NNR).T.detach().numpy(),cmap="jet")
    axs[2,0].set_title(r"$|\mathcal{R}(\hat{\rho}_t) - \mathcal{R}(\rho_t)|$")
    axs[2,0].set_xlabel(r"$t$")
    axs[2,0].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[0,1].imshow(rho_NNI.detach().numpy().T,cmap="jet")
    axs[0,1].set_title(r"$\mathcal{I}(\rho_{NN})$")
    axs[0,1].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[1,1].imshow(rho_data.imag.T.detach().numpy(),cmap="jet")
    axs[1,1].set_title(r"$\mathcal{I}(\hat{\rho}_t)$")
    axs[1,1].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im = axs[2,1].imshow(abs(rho_data.imag-rho_NNI).T.detach().numpy() ,cmap="jet")
    axs[2,1].set_title(r"$|\mathcal{I}(\hat{\rho}_{NN}) - \mathcal{I}(\rho_t)|$")
    axs[2,1].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    plt.tight_layout()
    plt.show()

class SIN(nn.Module):
    def __init__(self): 
        super(SIN, self).__init__() 
    def forward(self, x):
        return tc.sin(x)
    
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

class Train:
    def __init__(self,N_qubit, device="cpu",Base_do_jc=None):
        self.device     = device
        self.N_qubit    = N_qubit
        
        if N_qubit ==0 :
            self.base_rho  = Base_do_jc
        else:
            self.base_rho   = 2**N_qubit

    def _creat_data(self,J,dissi,N,tfinal,Observavel,all_data=None):
        
        if self.N_qubit == 0:
            self.rho_data, self.valor_esperado_data,self.hamiltonina_data,self.Observavel_data,self.Lindblad_data  = data_jc(
                lista_J = J,
                dissipation = dissi,
                tfinal  = tfinal,
                N       = N,
                device  = self.device,
                Observavel=Observavel) 
        elif self.N_qubit  == 1:
            self.rho_data, self.valor_esperado_data,self.hamiltonina_data,self.Observavel_data,self.Lindblad_data  = data_qubit_one(
                lista_J = J,
                dissipation = dissi,
                tfinal  = tfinal, 
                N       = N,
                device  = self.device,
                Observavel=Observavel)
        elif self.N_qubit  == 2:
            self.rho_data, self.valor_esperado_data,self.hamiltonina_data,self.Observavel_data,self.Lindblad_data  = data_qubit_two(
                lista_J = J,
                dissipation = dissi,
                tfinal  = tfinal, 
                N       = N,
                device  = self.device,
                Observavel=Observavel)
        else:
            self.rho_data, self.valor_esperado_data,self.hamiltonina_data,self.Observavel_data,self.Lindblad_data = all_data

    def _prepare_input(self,lista_J,lista_dissi,tfinal=None,N=None,observavel=0,data_=None,lista_time=None):
        self.lista_J    = lista_J
        self.dissipation = lista_dissi    
        self.observavel = observavel  
        if lista_time == None:
            self.tfinal_    = tfinal
            self.N          = N
                        
            self.t_train = tc.linspace(
                0,
                self.tfinal_,
                self.N,
                dtype   = tc.float32,
                requires_grad = True,
                device  = self.device).reshape((-1, 1)).to(self.device)
        else:
            self.t_train = lista_time.to(self.device)
            self.tfinal_    = np.float64(lista_time[-1].detach().numpy())
            self.N          = len(lista_time)
        self._creat_data(
            J      = lista_J.cpu().detach().numpy(),
            dissi  = lista_dissi.cpu().detach().numpy(),
            N      = self.N ,
            tfinal = self.tfinal_,
            Observavel = self.observavel,
            all_data = data_)

    def _initialize_networks(self,neuronio,funçao_de_ativa=None,path=".../", load_net=False,dropout_prob=0.0):
        
        self.N_neuronio = neuronio
        
        if funçao_de_ativa==None:
            funçao_de_ativa = [SIN()]*len(self.N_neuronio)
        
        self.real_net   = Rede(
            neuronio    = neuronio,
            input_      = 1 + len(self.lista_J) ,
            output_     = self.base_rho**2,
            activation  = funçao_de_ativa,
            dropout_prob=dropout_prob
            ).to(self.device)

        self.imag_net   = Rede(
            neuronio    = neuronio,
            input_      = 1+ len(self.lista_J) ,
            output_     = self.base_rho**2,
            activation  = funçao_de_ativa,
            dropout_prob=dropout_prob
            ).to(self.device)
    
        if load_net == True:
            self.real_net = tc.load(path)['real_net']
            self.imag_net = tc.load(path)['imag_net']
            self.real_net.to(device=self.device)
            self.imag_net.to(device=self.device)  

    def _initialize_optimizer(self,lr=0.01, step_size=500, gamma=0.9, epocas=5000,weight_decai=0,path=".../", load_net=False):
        
        self.epocas = epocas
        self.LOSS   = []

        self.opt = tc.optim.Adam(
            list(self.real_net.parameters()) + list(self.imag_net.parameters()),#+ list(self.Mask.parameters()),
            lr = lr,
            amsgrad=True ,
            weight_decay =weight_decai)
        
        self.scheduler = StepLR(
            self.opt,
            step_size = step_size,
            gamma   = gamma)
        
        if load_net == True:
            self.opt.load_state_dict(tc.load(path)['otimization'])
            self.scheduler.load_state_dict(tc.load(path)['step_lr'])
            self.LOSS = tc.load(path)['loss']
            
    def plot_loss(self):
        plt.subplots(figsize=(4, 4))
        plt.plot(self.LOSS)
        plt.yscale("log")
        plt.show()

    def plot_evaluate(self,lista_J,lista_dissi,N=None,Observavel=0,save_plot=False,lista_time=None,index_i=0,index_f=-1):
        self.real_net.eval()
        self.imag_net.eval()
               
        ####### inputs #######
        if lista_time == None:
            t_train  = tc.linspace(0,self.tfinal_,N).reshape((-1, 1))
            final_  = inputs[-1]
        else:
            t_train  = lista_time
            N       = len(lista_time)
            final_  = np.float64(lista_time[-1].detach().numpy())
            
        ####### inputs #######
        deltalista_ = self.lista_J.repeat((self.N,1))
        inputs      = tc.cat((t_train, deltalista_),axis=1)
        
        ####### Forward pass #######
        y_r = self.real_net(inputs)
        y_i = self.imag_net(inputs)
        
        ####### Criando rho no farmato de matriz #######
        rho = y_r + 1j * y_i
        rho = rho.reshape((self.N, self.base_rho, self.base_rho))  
           
        ####### Data #######
        self._creat_data(
            J           = lista_J.cpu().detach().numpy(),
            dissi       = lista_dissi.cpu().detach().numpy(),
            N           = N,
            tfinal      = final_,
            Observavel  = Observavel)

        plots_rho(
            rho.real.reshape((self.N, self.base_rho**2))[index_i:index_f],
            rho.imag.reshape((self.N, self.base_rho**2))[index_i:index_f],
            self.rho_data[index_i:index_f])
        
        expected_plot(
            rho_    = rho[index_i:index_f],
            O_      = self.Observavel_data,
            expected_data = self.valor_esperado_data[index_i:index_f],
            time_   = t_train[index_i:index_f],
            save_plot=save_plot)

    def train(self):
        for _ in tqdm(range(self.epocas)):
            self._train_epoch()

    def _train_epoch(self):
        ####### inputs #######
        deltalista_ = self.lista_J.repeat((self.N,1))
        inputs      = tc.cat((self.t_train, deltalista_),axis=1)
        
        ####### Forward pass #######
        y_r = self.real_net(inputs)
        y_i = self.imag_net(inputs)
        
        ####### Criando rho no farmato de matriz #######
        #y_i = M*y_i
        rho = y_r + 1j * y_i
        #rho = M*rho
        rho = rho.reshape((self.N, self.base_rho, self.base_rho))     
        
        ####### Los edo #######
        loss_edo = Loss_EDO(
            H_      = self.hamiltonina_data,
            rho_r   = rho.real.reshape((self.N, self.base_rho**2)),
            rho_i   = rho.imag.reshape((self.N, self.base_rho**2)),
            O_      = self.Lindblad_data,
            tempo   = self.t_train,
            baserho_= self.base_rho)

        ####### loss IC #######
        index   = tc.where(self.t_train ==0)[0]

        ####### loss IC #######
        loss_ic = msa_loss(
            rho[index].reshape((1, self.base_rho**2)),
            self.rho_data[0])
        
        ####### loss data(expected values) #######
        # loss_data   = 0
        # for i in range(len(self.Observavel_data)):
        #     v_esperados = expected(rho, self.Observavel_data[i]).sum(dim=-1).real
        #     loss_data += msa_loss(v_esperados[i], self.valor_esperado_data[0, i]) 
        index       = tc.arange(0, 100, dtype=tc.int)
        Tr_rho_O    = expected(rho[index], self.Observavel_data).sum(dim=-1).view(-1,1)
        
        loss_data   = 0
        for i in range(len(Tr_rho_O)):
            loss_data += msa_loss(Tr_rho_O[i], self.valor_esperado_data[index, i]) 
            
        ####### Termo de normalizaçao #######
        Tr_rho      = diagonal(rho).sum(-1)
        loss_norma  = msa_loss(Tr_rho,1)

        ####### Calculate the loss #######
        loss = loss_norma + loss_ic + loss_edo + loss_data

        ####### Backpropagation #######
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.scheduler.step()

        self.LOSS.append(loss.cpu().detach().numpy())
    
    def save_net(self, path,cond="all"):
        if cond == "all":
            tc.save({"real_net": self.real_net,
                     "imag_net": self.imag_net,
                     "otimization":self.opt.state_dict(),
                     "step_lr": self.scheduler.state_dict(),
                     "loss":self.LOSS},path)
        else :
            tc.save({"real_net": self.real_net.state_dict(),
                    "imag_net": self.imag_net.state_dict(),
                    "otimization":self.opt.state_dict(),
                    "step_lr": self.scheduler.state_dict(),
                    "loss":self.LOSS},path)     
          
class Train_simetric(Train):
    def __init__(self,N_qubit, device="cpu",Base_do_jc=None):
        super().__init__(N_qubit, device,Base_do_jc)
        self.device   = device
        self.N_qubit  = N_qubit
        
        if N_qubit ==0 :
            self.base_rho = Base_do_jc
        else:
            self.base_rho   = 2**N_qubit

    def _initialize_networks(self,neuronio ,funçao_de_ativa=None,path=".../", load_net=False):
        self.N_neuronio = neuronio
        if funçao_de_ativa==None:
            funçao_de_ativa = [SIN()]*len(self.N_neuronio)

        self.real_net = Rede(
            neuronio   = neuronio ,
            input_     = len(self.lista_J[0])+len(self.dissipation[0])+1,
            output_    = self.base_rho**2,
            activation = funçao_de_ativa
            ).to(self.device)
        
        if load_net == True:
            self.real_net = tc.load(path)['real_net']
            self.real_net.to(device=self.device)

    def _initialize_optimizer(self,lr,step_size ,gamma ,epocas,path=".../", load_net=False):
        self.epocas = epocas
        self.LOSS   = []
        
        self.opt = tc.optim.Adam(
            self.real_net.parameters(),
            lr = lr,
            amsgrad = True,
        )
        self.scheduler = StepLR(self.opt, step_size=step_size, gamma=gamma)
        if load_net == True:
            self.opt.load_state_dict(tc.load(path)['otimization'])
            self.scheduler.load_state_dict(tc.load(path)['step_lr'])
            self.LOSS = tc.load(path)['loss']

    def plot_evaluate(self,lista_J,lista_dissi,N,Observavel,save_plot=False,index_i=0,index_f=-1):
        self.real_net.eval()

        ####### inputs #######
        time        = tc.linspace(0,self.tfinal_,N).reshape((-1, 1))
        deltalista_ = lista_J.repeat((N,1))
        disslista_  = lista_dissi.repeat((N,1))
        inputs      = tc.cat((time,deltalista_, disslista_),axis=1)

        ####### Forward pass #######
        saida = self.real_net(inputs)

        ####### Data #######
        self._creat_data(
            J           = lista_J.cpu().detach().numpy(),
            dissi       = lista_dissi.cpu().detach().numpy(),
            N           = N,
            tfinal      = self.tfinal_,
            Observavel  = Observavel)      

        ### Re-estruturando a saida da rede ###
        if self.N_qubit == 1:
            ### Re-estruturando a saida da rede ###
            a_  = saida[:,0].view(-1,1)
            b_  = (saida[:,1] +1j*saida[:,2] ).view(-1,1)
            c_  = tc.conj_physical(b_)
            d_  = saida[:,3].view(-1,1)
            rho = tc.cat([a_,b_,c_,d_],dim=1) # Está no formado [self.N, 4] vetor para cada instante de tempo
        elif self.N_qubit == 2:
            ### Re-estruturando a saida da rede ###
            # a , b , c , d
            # b , f , g , h
            # c , g , l , m
            # d , h , m , n

            a_  =  saida[:,0].view(-1,1)
            b_  = (saida[:,1] +1j*saida[:,2] ).view(-1,1)
            c_  = (saida[:,3] +1j*saida[:,4] ).view(-1,1) #tc.conj_physical(b_)
            d_  = (saida[:,5] +1j*saida[:,6] ).view(-1,1)

            e_  = tc.conj_physical(b_)
            f_  =  saida[:,7].view(-1,1)
            g_  = (saida[:,8] +1j*saida[:,9] ).view(-1,1)
            h_  = (saida[:,10] +1j*saida[:,11] ).view(-1,1)

            i_  = tc.conj_physical(c_)
            j_  = tc.conj_physical(g_)
            l_  =  saida[:,12].view(-1,1)
            m_  = (saida[:,13] +1j*saida[:,14] ).view(-1,1)

            n_  = tc.conj_physical(d_)
            o_  = tc.conj_physical(h_)
            p_  = tc.conj_physical(m_)
            q_  =  saida[:,15].view(-1,1)

            rho = tc.cat([a_,b_,c_,d_,e_,f_,g_,h_,i_,j_,l_,m_,n_,o_,p_,q_],dim=1)

        plots_rho(rho.real, rho.imag, self.rho_data)
        expected_plot(rho_ = rho.reshape((N, self.base_rho, self.base_rho)),
                        O_ = self.Observavel_data,
                        expected_data = self.valor_esperado_data ,
                        time_= time,
                        save_plot=save_plot)

    def _train_epoch(self,lista_J,lista_dissi):

        ####### inputs #######
        deltalista_ = lista_J.repeat((self.N,1))
        disslista_  = lista_dissi.repeat((self.N,1))
        inputs      = tc.cat((self.t_train, deltalista_, disslista_),axis=1)

        ####### Data #######
        self._creat_data(
            J           = lista_J.cpu().detach().numpy(),
            dissi       = lista_dissi.cpu().detach().numpy(),
            N           = 3,
            tfinal      = self.tfinal_,
            Observavel  = self.observalvel_op)

        ####### Forward pass #######
        saida = self.real_net(inputs)

        ### Re-estruturando a saida da rede ###
        if self.N_qubit == 1:
            ### Re-estruturando a saida da rede ###
            a_  = saida[:,0].view(-1,1)
            b_  = (saida[:,1] +1j*saida[:,2] ).view(-1,1)
            c_  = tc.conj_physical(b_)
            d_  = saida[:,3].view(-1,1)
            rho = tc.cat([a_,b_,c_,d_],dim=1) # Está no formado [self.N, 4] vetor para cada instante de tempo
        elif self.N_qubit == 2:
            ### Re-estruturando a saida da rede ###
            # a , b , c , d
            # b , f , g , h
            # c , g , l , m
            # d , h , m , n

            a_  =  saida[:,0].view(-1,1)
            b_  = (saida[:,1] +1j*saida[:,2] ).view(-1,1)
            c_  = (saida[:,3] +1j*saida[:,4] ).view(-1,1) #tc.conj_physical(b_)
            d_  = (saida[:,5] +1j*saida[:,6] ).view(-1,1)

            e_  = tc.conj_physical(b_)
            f_  =  saida[:,7].view(-1,1)
            g_  = (saida[:,8] +1j*saida[:,9] ).view(-1,1)
            h_  = (saida[:,10] +1j*saida[:,11] ).view(-1,1)

            i_  = tc.conj_physical(c_)
            j_  = tc.conj_physical(g_)
            l_  =  saida[:,12].view(-1,1)
            m_  = (saida[:,13] +1j*saida[:,14] ).view(-1,1)

            n_  = tc.conj_physical(d_)
            o_  = tc.conj_physical(h_)
            p_  = tc.conj_physical(m_)
            q_  =  saida[:,15].view(-1,1)

            rho = tc.cat([a_,b_,c_,d_,e_,f_,g_,h_,i_,j_,l_,m_,n_,o_,p_,q_],dim=1)

        ####### Los edo #######
        loss_edo = Loss_EDO(    
            H_      = self.hamiltonina_data,
            rho_r   = rho.real,
            rho_i   = rho.imag,
            O_      = self.Lindblad_data,
            tempo   = inputs,
            baserho_= self.base_rho)

        ####### loss IC #######
        loss_ic  = mse_loss(rho[0].real, self.rho_data[0].real)
        loss_ic += mse_loss(rho[0].imag, self.rho_data[0].imag)

        ####### Criando rho no farmato de matriz #######
        rho     = rho.reshape((self.N, self.base_rho, self.base_rho))
        
        ####### loss data(expected values) #######
        Tr_rho_O    = expected(rho[0],self.Observavel_data).sum(dim=-1).view(-1,1)
        loss_data   = 0   
        for i in range(len(Tr_rho_O)):
            loss_data += msa_loss(Tr_rho_O[i], self.valor_esperado_data[0, i]) 

        ####### Termo de normalizaçao #######
        Tr_rho      = diagonal(rho).sum(-1)
        loss_norma  = msa_loss(Tr_rho,1)

        ####### Calculate the loss #######
        loss = loss_norma + loss_ic + loss_edo + loss_data

        ####### Backpropagation #######
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.scheduler.step()

        self.LOSS.append(loss.cpu().detach().numpy())

    def save_net(self, path,cond="all"):
        if cond == "all":
            tc.save({"real_net": self.real_net,
                     "otimization":self.opt.state_dict(),
                     "step_lr": self.scheduler.state_dict(),
                     "loss":self.LOSS},path)
        else :
            tc.save({"real_net": self.real_net.state_dict(),
                    "otimization":self.opt.state_dict(),
                    "step_lr": self.scheduler.state_dict(),
                    "loss":self.LOSS},path)