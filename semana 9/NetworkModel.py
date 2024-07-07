import numpy as np

class NetworkModel:
    def __init__(self, usernumber=100, RBnumber=10, total_model_params=101770, user_P=0.01, user_Bw=1, bs_P=1, N=10**-20):
        """
        Inicializa a simulação de aprendizado federado.

        Parâmetros:
        usernumber (int): Número de usuários.
        RBnumber (int): Número de blocos de recursos.
        total_model_params (int): Número total de parâmetros do modelo.
        user_P (float): Potência do usuário.
        user_Bw (int): Largura de banda do usuário em MHz.
        bs_P (int): Potência da base station em watts.
        N (float): Ruído térmico.
        """
        self.usernumber = usernumber
        self.RBnumber = RBnumber
        self.total_model_params = total_model_params
        self.user_P = user_P
        self.user_Bw = user_Bw
        self.bs_P = bs_P
        self.N = N
        
        self.user_I = self.calculate_user_I()
        self.d = self.generate_user_distances()
        self.user_SINR = self.calculate_user_SINR()
        self.user_rate = self.calculate_user_rate()
        self.bs_SINR = self.calculate_bs_SINR()
        self.bs_rate = self.calculate_bs_rate()
        self.user_delay = self.calculate_user_delay()
        self.bs_delay = self.calculate_bs_delay()
        self.totaldelay = self.user_delay + self.bs_delay
        self.user_energy_training = self.calculate_user_energy_training()
        self.user_upload_energy = self.calculate_user_upload_energy()
        self.total_energy = self.user_energy_training + self.user_upload_energy


    def calculate_user_I(self):
        """
        Calcula a interferência do usuário sobre os blocos de recursos.

        Retorna:
        np.array: Interferência de cada usuário.
        """
        i = np.array([0.05 + i * 0.01 for i in range(self.RBnumber)])
        user_I = (i - 0.04) * 0.000001
        return user_I


    def generate_user_distances(self):
        """
        Gera distâncias aleatórias dos usuários à base.

        Retorna:
        np.array: Distâncias dos usuários.
        """
        # np.random.seed(1)
        a, b = 100, 500
        d = a + (b - a) * np.random.rand(self.usernumber, 1)
        return d


    def calculate_user_SINR(self):
        """
        Calcula a relação sinal-interferência-ruído (SINR) dos usuários.

        Retorna:
        np.array: SINR de cada usuário.
        """
        o = 1  # Parâmetro de desvanecimento de Rayleigh
        h = o * (self.d ** (-2))
        user_SINR = self.user_P * h / (self.user_I + self.user_Bw * self.N)
        return user_SINR


    def calculate_user_rate(self):
        """
        Calcula a taxa de transmissão dos usuários.

        Retorna:
        np.array: Taxa de transmissão de cada usuário em Mbps.
        """
        return self.user_Bw * np.log2(1 + self.user_SINR)


    def calculate_bs_SINR(self):
        """
        Calcula a relação sinal-interferência-ruído (SINR) da base station.

        Retorna:
        np.array: SINR da base station.
        """
        bs_I = 0.06 * 0.000003  # Interferência sobre o downlink
        bs_Bw = 20  # MHz
        h = 1 * (self.d ** (-2))  # Recalcula h como não está armazenado separadamente
        bs_SINR = self.bs_P * h / (bs_I + self.N * bs_Bw)
        return bs_SINR


    def calculate_bs_rate(self):
        """
        Calcula a taxa de transmissão da base station.

        Retorna:
        np.array: Taxa de transmissão da base station em Mbps.
        """
        bs_Bw = 20  # MHz
        return bs_Bw * np.log2(1 + self.bs_SINR)


    def calculate_user_delay(self):
        """
        Calcula o atraso dos usuários.

        Retorna:
        np.array: Atraso dos usuários em segundos.
        """
        Z = self.total_model_params * 4 / (1024 ** 2)  # MBytes
        return Z / self.user_rate


    def calculate_bs_delay(self):
        """
        Calcula o atraso da base station.

        Retorna:
        np.array: Atraso da base station em segundos.
        """
        Z = self.total_model_params * 4 / (1024 ** 2)  # MBytes
        return Z / self.bs_rate


    def calculate_user_energy_training(self):
        """
        Calcula a energia usada para treinar o modelo pelos usuários.

        Retorna:
        float: Energia de treinamento dos usuários em joules.
        """
        energy_coeff = 10 ** (-27)
        cpu_cycles = 40
        cpu_freq = 10 ** 9
        Z = self.total_model_params * 4 / (1024 ** 2)  # MBytes
        return energy_coeff * cpu_cycles * (cpu_freq ** 2) * Z


    def calculate_user_upload_energy(self):
        """
        Calcula a energia usada pelos usuários para upload.

        Retorna:
        np.array: Energia de upload dos usuários em joules.
        """
        return self.user_P * self.user_delay


    def selecao_clientes_aleatoria(self):
        """
        Seleciona clientes aleatoriamente para a simulação.

        Retorna:
        tuple: Array de atribuição e clientes selecionados.
        """
        # np.random.seed(1)
        assignment = np.zeros(self.usernumber, dtype=int)

        if self.RBnumber < self.usernumber:
            assignment[np.random.permutation(self.usernumber)[:self.RBnumber]] = 1
        else:
            assignment[:] = 1

        selected_clients = np.where(assignment > 0)[0]
        return assignment, selected_clients


    def atribuicao_RBs(self, selected_clients):
        """
        Atribui blocos de recursos (RBs) aos usuários selecionados usando heurística baseada na distância.

        Parâmetros:
        selected_clients (np.array): Clientes selecionados.

        Retorna:
        tuple: Array de atribuição de RBs e clientes selecionados reordenados.
        """
        qassignment = np.zeros(self.usernumber, dtype=int)
        combined_data = list(zip(self.d[selected_clients], np.arange(len(selected_clients))))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        rb_allocation, pos_list = zip(*sorted_data)
        qassignment[selected_clients[np.array(pos_list)]] = 1
        return qassignment, selected_clients[np.array(pos_list)]


    def calculate_final_total_energy(self, selected_clients, sender_clients, rb_allocation):
        """
        Calcula a energia total final para os clientes selecionados.

        Parâmetros:
        selected_clients (np.array): Clientes selecionados.
        rb_allocation (np.array): Alocação de RBs.

        Retorna:
        float: Energia total final em joules.
        """
        final_total_energy = 0
        for i in range(len(selected_clients)):
            iclient = selected_clients[i]
            irb = rb_allocation.tolist().index(iclient)

            if iclient in sender_clients:
                upload_energy = self.user_upload_energy[iclient, irb]
            else:
                upload_energy = 0
            
            final_total_energy += self.user_energy_training + upload_energy
                
            print(f"{i}: -> Disp: {selected_clients[i]}, RB: {irb}: {self.total_energy[iclient, irb]}")
        return final_total_energy


    def run_simulation(self):
        """
        Executa a simulação completa de aprendizado federado.

        Retorna:
        float: Energia total final em joules.
        """
        assignment, selected_clients = self.selecao_clientes_aleatoria()
        qassignment, rb_allocation = self.atribuicao_RBs(selected_clients)
        final_total_energy = self.calculate_final_total_energy(selected_clients, rb_allocation)
        return final_total_energy
