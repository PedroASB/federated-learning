# Aprendizado Federado
## Introdução

Este projeto aborda a implementação e avaliação de uma simulação de aprendizado federado, utilizando a base de dados MNIST. O objetivo é explorar diferentes estratégias de transmissão de parâmetros de modelos treinados localmente pelos clientes, otimizando a comunicação e os recursos computacionais no ambiente federado. As variantes de transmissão incluem uma abordagem de transmissão total, uma transmissão condicional baseada na diferença dos pesos, e uma transmissão aleatória. O projeto também incorpora a simulação de um ambiente de rede com parâmetros fixos e variáveis para avaliar a eficiência da comunicação e o consumo de energia. A implementação detalhada inclui classes para a modelagem da rede, do cliente e do servidor, além de funções específicas para seleção de clientes, alocação de canais de uplink, e agregação de modelos. A execução do sistema envolve a coordenação de múltiplas rodadas de treinamento federado, análise do desempenho do modelo global, e monitoramento de métricas de acurácia, perda, e consumo de energia, culminando na visualização gráfica dos resultados obtidos.

---

## Desenvolvedores
- Nickolas Carlos Carvalho Silva ([nickolascarlos](<https://github.com/nickolascarlos>))
- Pedro Augusto Serafim Belo ([PedroASB](<https://github.com/PedroASB>))

---

## Dataset

Para este projeto, foi escolhida a base de dados MNIST, amplamente utilizada na avaliação de modelos de aprendizado de máquina. Esta base é composta por 70.000 imagens em escala de cinza, com dimensões de 28x28 pixels, de dígitos manuscritos. As imagens são divididas em 60.000 para treinamento e 10.000 para teste.

---

## Modelo de Rede Neural

O modelo de rede neural `ModelMLP` define uma rede neural perceptron multicamadas (MLP) utilizando a biblioteca TensorFlow/Keras. A função estática `create_model` cria e retorna uma instância do modelo sequencial, composta por duas camadas densas. A primeira camada densa possui 128 neurônios com a função de ativação ReLU e uma forma de entrada de 784, ideal para dados de entrada como imagens de 28x28 pixels. A segunda camada é uma camada densa com 10 neurônios e utiliza a função de ativação softmax, adequada para problemas de classificação com 10 classes. O modelo é compilado com o otimizador Adam e usa a função de perda de entropia cruzada esparsa categórica, além de ser configurado para monitorar a métrica de precisão durante o treinamento.

```python
# Modelo de Rede Neural
class ModelMLP:
    @staticmethod
    def create_model():
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dense(10, activation="softmax")
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        return model
```



---

## Modelo de Rede Sem Fio

A classe `NetworkModel` é projetada para simular um ambiente de aprendizado federado considerando diversos parâmetros e condições de rede. Entre os parâmetros pré-definidos estão a potência do usuário (`user_P`), a largura de banda do usuário (`user_Bw`), a potência da estação base (`bs_P`), e o ruído térmico (`N`). Estes parâmetros são fixos para todos os usuários, garantindo que a única variável que influencie a comunicação seja a distância entre os dispositivos e a estação base. A classe também define a quantidade de usuários (`usernumber`), o número de Resource Blocks (RBs) (`RBnumber`), e o número total de parâmetros do modelo (`total_model_params`).

Dentro da classe, funções são implementadas para calcular a interferência dos usuários (`calculate_user_I`), gerar distâncias aleatórias dos usuários até a estação base (`generate_user_distances`), e calcular a relação sinal-interferência-ruído (SINR) tanto para os usuários quanto para a estação base (`calculate_user_SINR`, `calculate_bs_SINR`). As taxas de transmissão são calculadas com base no SINR (`calculate_user_rate`, `calculate_bs_rate`), e os atrasos de comunicação são determinados a partir dessas taxas (`calculate_user_delay`, `calculate_bs_delay`). A energia necessária para o treinamento do modelo e para o upload dos dados também é estimada (`calculate_user_energy_training`, `calculate_user_upload_energy`).

Além disso, a classe oferece métodos para selecionar aleatoriamente os clientes que participarão de cada rodada de comunicação (`selecao_clientes_aleatoria`) e para atribuir canais de uplink a esses clientes com base em uma heurística (`atribuicao_RBs`). A função `calculate_final_total_energy` então computa a energia total final considerando os clientes selecionados e suas respectivas alocações de RBs. O modelo assume um cenário próximo ao ideal, sem contabilizar perdas de pacotes, focando na eficiência da comunicação influenciada predominantemente pelas distâncias dos usuários à estação base.

### Seleção de clientes

```python
def selecao_clientes_aleatoria(self):
	assignment = np.zeros(self.usernumber, dtype=int)
	
	if self.RBnumber < self.usernumber:
	    assignment[np.random.permutation(self.usernumber)[:self.RBnumber]] = 1
	else:
	    assignment[:] = 1
	
	selected_clients = np.where(assignment > 0)[0]
	return assignment, selected_clients
```

A seleção dos clientes para cada rodada de comunicação é feita aleatoriamente usando a função `selecao_clientes_aleatoria()` da classe NetworkModel. Essa função realiza a seleção de clientes e retorna essa seleção por meio de duas representações distintas: uma matriz de atribuição, em que o os elementos com valor igual a 1 correspondem aos índices dos clientes selecionados, e a lista de índices de clientes selecionados equivalente.

Inicialmente, essa função cria uma lista de atribuição preenchida com zeros, do tamanho igual ao número total de clientes. Depois, verifica se o número de Resource Blocks (RBs) disponíveis é menor que o número de clientes. Se for o caso, a função seleciona aleatoriamente um número de clientes igual ao número de RBs, embaralhando os índices dos clientes e escolhendo os primeiros clientes até o número de RBs. Se o número de RBs for maior ou igual ao número de clientes, todos os clientes são selecionados. A função então identifica os índices dos clientes selecionados onde o valor na lista de atribuição é maior que zero, e retorna o array de atribuição de clientes e a lista com os índices dos clientes selecionados.

### Atribuição de canais de uplink

```python
def atribuicao_RBs(self, selected_clients):
  qassignment = np.zeros(self.usernumber, dtype=int)
	combined_data = list(zip(self.d[selected_clients], np.arange(len(selected_clients))))
	sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
	rb_allocation, pos_list = zip(*sorted_data)
	qassignment[selected_clients[np.array(pos_list)]] = 1
	return qassignment, selected_clients[np.array(pos_list)]
```

A função `atribuicao_RBs` presente no modelo de rede define os canais de uplink para cada usuário, assegurando que a comunicação entre os dispositivos dos usuários e a estação base (BS) seja eficiente e confiável. O uplink refere-se à transmissão de dados do dispositivo do usuário para a estação base, essencial para enviar atualizações do modelo treinado localmente para um servidor central. A função utiliza uma heurística baseada na distância dos usuários à BS, priorizando aqueles mais distantes para garantir que recebam os recursos necessários devido à maior atenuação do sinal. Inicialmente, ela cria uma lista combinando as distâncias dos usuários selecionados e seus respectivos índices. Em seguida, essa lista é ordenada em ordem decrescente pela distância, garantindo que os usuários mais distantes sejam tratados primeiro. Os índices ordenados são então utilizados para atualizar um array de atribuição, marcando os usuários que receberão os RBs. Assim, ao alocar de forma inteligente os Resource Blocks (RBs), que são as unidades básicas de frequência e tempo, a função otimiza a comunicação ao assegurar que os usuários com maiores desafios de conectividade recebam prioridade na alocação de recursos, melhorando assim a eficiência e a estabilidade da rede.

### Cálculo da energia total

```python
def calculate_final_total_energy(self, selected_clients, sender_clients, rb_allocation):
    final_total_energy = 0
    for i in range(len(selected_clients)):
        iclient = selected_clients[i]
        irb = rb_allocation.tolist().index(iclient)
        if iclient in sender_clients:
            upload_energy = self.user_upload_energy[iclient, irb]
        else:
            upload_energy = 0
        final_total_energy += self.user_energy_training + upload_energy
    return final_total_energy
```

O cálculo da energia total é realizado somando a energia de treinamento e a energia de upload para cada cliente selecionado. Para cada cliente, é identificado qual recurso de banda (RB) está alocado para ele. Se o cliente está na lista de clientes que enviam dados (`sender_clients`), a energia de upload específica para esse cliente e RB é somada à energia de treinamento. Caso contrário, apenas a energia de treinamento é considerada. Esse processo é repetido para todos os clientes selecionados, e a soma dessas energias é acumulada na variável `final_total_energy`, que é então retornada como o resultado final em joules.

---

## Aprendizado Federado

### Classe do cliente

A classe `Client` representa um cliente individual no ambiente de aprendizado federado. Cada cliente possui um modelo MLP local que é treinado usando um subconjunto dos dados MNIST específico para aquele cliente. A classe `Client` gerencia o carregamento dos dados de treinamento e teste, bem como a inicialização e configuração do modelo. Além disso, a classe `Client` implementa métodos para realizar o treinamento local dos dispositivos e então transmitir os novos parâmetros para o servidor, e também para avaliar o desempenho do modelo localmente, proporcionando métricas de acurácia e perda que ajudam a monitorar o progresso do treinamento e a qualidade do modelo ao longo das rodadas de aprendizado federado.

Existem três variantes de classes `Client`, em que uma sempre realiza a transmissão de seus parâmetros em uma rodada de treinamento, outra em que há uma condição imposta para tal transmissão seja feita, e uma última em que em a decisão sobre a transmissão é feita de forma aleatória. Essas variantes foram implementadas em `MLP Transmissão Total.ipynb`, `MLP  Transmissão Condicional.ipynb` e `MLP Transmissão Aleatória.ipynb`, respectivamente.

**Transmissão total:**
Esta é a versão básica do modelo de aprendizado federado, em que o cliente sempre transmite seus parâmetros atualizados após cada rodada de treinamento. A função `fit` carrega os pesos recebidos, treina o modelo localmente e, em seguida, envia os novos pesos ao servidor central. Esta abordagem faz com que o modelo global seja atualizado com as contribuições de todos os clientes em cada rodada.

```python
def fit(self, parameters, config=None):
	self.model.set_weights(parameters)
	history = self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=128, validation_data=(self.x_test, self.y_test), verbose=False)
	sample_size = len(self.x_train)
	print(f"Acurácia: {history.history['val_accuracy'][-1]} | ")
	return self.model.get_weights(), sample_size, {"val_accuracy": history.history['val_accuracy'][-1], 
	                                               "val_loss": history.history['val_loss'][-1]}
```

**Transmissão condicional:**
Nesta variante proposta, a transmissão dos parâmetros é realizada apenas se a diferença entre os pesos atuais e os pesos anteriores for significativa. Após o treinamento, a função `fit` calcula a diferença percentual média entre os pesos novos e antigos. Se essa diferença for menor que um limiar definido (`EPSILON_DELTA`), o cliente decide não transmitir seus pesos, economizando largura de banda e recursos computacionais. Esta abordagem é útil para reduzir a comunicação desnecessária, especialmente quando as atualizações de peso são mínimas.

```python
def fit(self, parameters, config=None):
	self.model.set_weights(parameters)
	history = self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=128,
	                         validation_data=(self.x_test, self.y_test), verbose=False)
	self.current_weights = self.model.get_weights()
	sample_size = len(self.x_train)
	print(f"Acurácia: {history.history['val_accuracy'][-1]} | ", end='')
	if self.old_weights is not None:
	    weight_diff = np.mean([
	        np.mean(np.abs((w1 - w2) / w2) * 100) 
	        for w1, w2 in zip(self.current_weights, self.old_weights)
	        if np.sum(np.abs(w2)) > 0  # Evitar divisão por zero
	    ])
	    print(f"Diferença dos pesos: {weight_diff}")
	    self.old_weights = self.current_weights
	    if weight_diff < EPSILON_DELTA:
	        print("weight_diff < EPSILON_DELTA. O modelo não será transmitido.")
	        return None
	    else:
	        print("ENVIADO!")
	        return self.current_weights, sample_size, {"val_accuracy": history.history['val_accuracy'][-1],
	                                                       "val_loss": history.history['val_loss'][-1]}
	else:
	    print("ENVIANDO PRIMEIRA")
	    self.old_weights = self.current_weights
	    return self.current_weights, sample_size, {"val_accuracy": history.history['val_accuracy'][-1],
	                                                       "val_loss": history.history['val_loss'][-1]}
```

**Transmissão aleatória:**
Nesta versão, a decisão de transmitir os parâmetros é feita de forma aleatória. Após o treinamento, a função `fit` determina aleatoriamente, com base em um valor de probabilidade (`LAMBDA`), se os pesos serão enviados ao servidor. Esta abordagem tem como finalidade ajudar a verificar o custo-benefício do modelo de transmissão condicional. A título de exemplo, suponha que em uma transmissão condicional houveram 700 transmissões de um máximo de 1000, ou seja, 70% do máximo. Neste caso, testa-se uma transmissão aleatória em que os modelos locais atualizados serão transmitidos aproximadamente 70% das vezes, utilizando um critério simples de decisão (probabilidade), sem gastos com cálculos de diferença de pesos. Dessa forma, se forem obtidos resultados similares com uma transmissão aleatória, isso indicaria um mal custo-benefício do modelo com transmissão condicional; caso contrário, isso seria uma evidência da efetividade de se realizar o cálculo da diferença de pesos como critério de decisão.

```python
def fit(self, parameters, config=None):
	self.model.set_weights(parameters)
	history = self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=128, validation_data=(self.x_test, self.y_test), verbose=False)
	sample_size = len(self.x_train)
	print(f"Acurácia: {history.history['val_accuracy'][-1]} | ", end='')
	
	if random.random() > LAMBDA and self.old_weights is not None:
	    print("Modelo NÃO transmitido!")
	    return None
	
	print('Modelo transmitido!')
	self.old_weights = self.model.get_weights()
	return self.model.get_weights(), sample_size, {"val_accuracy": history.history['val_accuracy'][-1], 
	                                               "val_loss": history.history['val_loss'][-1]}
```

### Classe do servidor

A classe `Server` implementa um sistema de aprendizado federado, uma abordagem distribuída em que vários clientes colaboram para treinar um modelo centralizado sem compartilhar diretamente seus dados pessoais.

Inicialmente, o servidor carrega os conjuntos de dados de treino e teste do MNIST,  bem como os modelos de cada cliente participante do sistema, que são armazenados na lista `clients_model_list`.

No método `fit`, o servidor coordena o processo de treinamento federado selecionando os clientes e inicia o ajuste do modelo global (`w_global`). Cada cliente selecionado executa um processo de treinamento local em seu próprio conjunto de dados, resultando em um modelo atualizado e, em seguida, decide se envia ou não o modelo resultante de volta ao servidor. Se um novo modelo for enviado, o servidor o armazena e o utiliza na atualização do modelo global. Caso o cliente tenha optado por não realizar o envio do seu novo modelo, treinado durante a rodada corrente, ele retorna None para o servidor que, então, usará o último modelo enviado pelo cliente para realizar a atualização do modelo global.

```python
def fit(self):
	weight_list, sample_sizes_list, info_list, sender_clients = [], [], [], []
	for i, pos in enumerate(self.selected_clients):
	    print(f"Cliente #{pos+1} | ", end='')
	
	    # Se o cliente decidir não enviar o modelo, ele retorna um None
	    new_model = self.clients_model_list[pos].fit(parameters=self.w_global)
	
	    weights, size, info = new_model if new_model is not None else self.get_client_previous_model(pos)
	    weight_list.append(weights)
	    sample_sizes_list.append(size)
	    info_list.append(info)
	
	    # Se o cliente enviou um novo modelo, salve-o e insira-o ao sender_client
	    if new_model is not None:
	        self.save_client_previous_model(pos, (weights, size, info_list))
	        sender_clients.append(pos)
	
	return weight_list, sample_sizes_list, {"acc_loss_local":[(pos+1, info_list[i]) for i, pos in enumerate(self.selected_clients)]}, sender_clients
```

A função `aggregate_fit` é onde a lógica de agregação dos modelos locais dos clientes é efetivamente implementada. Ela calcula uma média ponderada dos pesos dos modelos recebidos, onde o peso de cada modelo é determinado pelo tamanho do conjunto de dados do respectivo cliente, de maneira que clientes com conjuntos de dados maiores contribuam mais significativamente para a atualização do modelo global a cada rodada.

```python
def aggregate_fit(self, weight_list, sample_sizes):
	self.w_global = []
	for weights in zip(*weight_list):
	    weighted_sum = 0
	    total_samples = sum(sample_sizes)
	    for i in range(len(weights)):
	        weighted_sum += weights[i] * sample_sizes[i]
	    self.w_global.append(weighted_sum / total_samples)
```

Após cada rodada de treinamento, o servidor avalia o desempenho do modelo global para monitoramento e análise, empregando, para tanto, o método `centralized_evaluation`, que calculará métricas de desempenho, como perda média e precisão, utilizando o conjunto de teste centralizado.

### Execução

A função `executar` é responsável por coordenar e gerenciar o processo de treinamento federado ao longo de várias rodadas. Ela inicia configurando o ambiente e os parâmetros necessários, incluindo a criação de clientes e a inicialização do servidor central. Em seguida, a função entra em um loop que se repete pelo número definido de rodadas. Em cada rodada, a função seleciona um subconjunto de clientes para treinar seus modelos localmente e, dependendo da diferença nos pesos, envia os modelos atualizados de volta ao servidor. O servidor, então, agrega os modelos recebidos para atualizar o modelo global. Após cada rodada, a função `executar` avalia o modelo global tanto de forma distribuída quanto centralizada, coletando métricas de desempenho como acurácia e perda. Além disso, a função registra informações detalhadas sobre as transmissões e o consumo de energia, permitindo uma análise abrangente da eficiência do processo de aprendizado federado. Ao final de todas as rodadas, são gerados gráficos que visualizam o desempenho do modelo ao longo do tempo, proporcionando insights valiosos sobre o comportamento e a eficácia da abordagem de aprendizado federado implementada.

## Resultados

### Transmissão Total

<p float="left">
    <img src="semana 9\graficos\transmissao_total\acuracia_transmissao_total.jpg" width="300" />
    <img src="semana 9\graficos\transmissao_total\energia_transmissao_total.jpg" width="310" /> 
    <img src="semana 9\graficos\transmissao_total\energia_acumulada_transmissao_total.jpg" width="295" />
</p>

---

### Transmissão Condicional (ε = 25)

<p float="left">
    <img src="semana 9\graficos\transmissao_condicional\epsilon_25\acuracia_transmissao_condicional.jpg" width="300" />
    <img src="semana 9\graficos\transmissao_condicional\epsilon_25\energia_transmissao_condicional.jpg" width="310" /> 
    <img src="semana 9\graficos\transmissao_condicional\epsilon_25\energia_acumulada_transmissao_condicional.jpg" width="290" />
</p>

### Transmissão Condicional (ε = 40)

<p float="left">
    <img src="semana 9\graficos\transmissao_condicional\epsilon_40\acuracia_transmissao_condicional.jpg" width="300" />
    <img src="semana 9\graficos\transmissao_condicional\epsilon_40\energia_transmissao_condicional.jpg" width="310" /> 
    <img src="semana 9\graficos\transmissao_condicional\epsilon_40\energia_acumulada_transmissao_condicional.jpg" width="290" />
</p>


### Transmissão Condicional (ε = 55)

<p float="left">
    <img src="semana 9\graficos\transmissao_condicional\epsilon_55\acuracia_transmissao_condicional.jpg" width="300" />
    <img src="semana 9\graficos\transmissao_condicional\epsilon_55\energia_transmissao_condicional.jpg" width="310" /> 
    <img src="semana 9\graficos\transmissao_condicional\epsilon_55\energia_acumulada_transmissao_condicional.jpg" width="290" />
</p>

---

### Transmissão Aleatória (≈ 300 transmissões)

<p float="left">
    <img src="semana 9\graficos\transmissao_aleatoria\validacao_epsilon_25\acuracia_transmissao_aleatoria.jpg" width="300" />
    <img src="semana 9\graficos\transmissao_aleatoria\validacao_epsilon_25\energia_transmissao_aleatoria.jpg" width="310" /> 
    <img src="semana 9\graficos\transmissao_aleatoria\validacao_epsilon_25\energia_acumulada_transmissao_aleatoria.jpg" width="295" />
</p>

### Transmissão Aleatória (≈ 400 transmissões)

<p float="left">
    <img src="semana 9\graficos\transmissao_aleatoria\validacao_epsilon_40\acuracia_transmissao_aleatoria.jpg" width="300" />
    <img src="semana 9\graficos\transmissao_aleatoria\validacao_epsilon_40\energia_transmissao_aleatoria.jpg" width="310" /> 
    <img src="semana 9\graficos\transmissao_aleatoria\validacao_epsilon_40\energia_acumulada_transmissao_aleatoria.jpg" width="295" />
</p>

### Transmissão Aleatória (≈ 250 transmissões)

<p float="left">
    <img src="semana 9\graficos\transmissao_aleatoria\validacao_epsilon_55\acuracia_transmissao_aleatoria.jpg" width="300" />
    <img src="semana 9\graficos\transmissao_aleatoria\validacao_epsilon_55\energia_transmissao_aleatoria.jpg" width="310" /> 
    <img src="semana 9\graficos\transmissao_aleatoria\validacao_epsilon_55\energia_acumulada_transmissao_aleatoria.jpg" width="295" />
</p>

