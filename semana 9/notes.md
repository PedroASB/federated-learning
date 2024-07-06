### Notes

- Artigo “Aprendizado Federado em Redes IoT sem Fio: Novo Algoritmo para a Seleção de Dispositivos e Alocação dos Recursos de Comunicação”
    - Modelo de comunicação
    - Consumo de energia
- Considerar mesma potência e mesma largura de banda (fixos) → o que irá influenciar é a distância
- Perda de pacotes não serão contabilizadas
- OFDM / OFDMA
    - Os primeiros canais têm menos interferência do que os últimos
- Ver sobre relação sinal-ruído
- Importante:
    - total_delay = user_upload_energy + user_energy_training
- Atribuição de canais de uplink para cada cliente
    - “`atribuicao_RBs_aleatoria`”: faz isso de forma aleatória
    - “`atribuicao_RBs`”: é uma heurística → devemos utilizar essa
- Tirar a seed das funções
- Gráfico de custo energético
    - https://github.com/LABORA-INF-U FG/FL-wDQN
- Ver:https://github.com/LABORA-INF-UFG/FL-wDQN/blob/main/ipynb/graphics-FL(Mnist)-MLP.ipynb
- Rodar cada epsilon 5x