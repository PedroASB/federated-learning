# Semana 4

- Estudo do artigo "Federated Learning Over Wireless IoT Networks
With Optimized Communication and Resources"
- A seguir, há uma visão geral do que foi estudado no artigo, no formato de resumo.

## Resumo

### **Modelo de Rede**

- Começando pelo modelo de rede, temos as seguintes definições:
- Dado um conjunto de clientes, definido pelo conjunto $\mathcal N$ que vai de $1$ até $N$, serão coletados dados desses clientes, assumindo que cada cliente $i$ possui um conjunto $\mathcal D_i$ de dados privados, que é definido por essa fórmula, onde $\xi$ é uma amostra de treinamento; e o dataset inteiro é representado por $\mathcal D$

### Aprendizado Federado

- Agora, com relação ao processo de aprendizado federado, temos o seguinte:
- Sabemos que no AF o objetivo é treinar um modelo global de machine learning por meio meio de dispositivos que treinam o modelo localmente e compartilham parâmetros, não dados. Nesse sentido, os autores definiram $w$ como sendo o modelo global, e o objetivo é minimizar uma função de perda $f(w)$
    - *[fórmula 1]*
    - Essa função envolve a função de perda de cada cliente $i$, $f_i(w)$, e isso é feito para cada amostra de treinamento desse cliente, e tudo isso para cada cliente
- A partir disso, baseado no algoritmo de média federada (FedAvg) aplica-se várias rodadas de comunicação para realizar o treinamento
    1. Primeiramente, em uma rodada $t$, o modelo global é enviado para todos os clientes *[ressaltar o fato de ser enviado para todos os clientes]*
    2. Em seguida, cada cliente treina seu modelo local aplicando um método que utiliza gradiente *[fórmula 2]*
    3. Por fim, tendo recebido os parâmetros dos modelos locais, o servidor os agrega para atualizar o modelo global *[fórmula 3]*

### **Modelo de Comunicação**

- Em seguida, os autores apresentam o modelo de comunicação, onde os clientes fazem upload seus modelos locais para o servidor por meio de acesso múltiplo por divisão de frequência (FDMA) *[fórmula 4]*
- Temos essa fórmula aqui em função da largura de banda alocada $(B^t_i)$ e potência de transmissão $(P^t_i)$ de um cliente $i$
- Temos também o tempo de comunicação de um cliente $i$ para o envio de um pacote $S$ do modelo local para o servidor, dado por *[fórmula 5]*
- E a probabilidade de interrupção da transmissão do modelo, dada por *[fórmula 6]*

### **Formulação do Problema**

- Uma vez que estamos tratando de uma rede sem fio realista, outros fatores têm de ser levados em conta além de simplesmente trazer um bom algoritmo de aprendizado federado
- Em geral, o processo de treinamento do AF escala o maior número possível de clientes em cada rodada de comunicação. No entanto, não é interessante que todos os clientes envolvidos no aprendizado transmitam seus novos modelos locais para o servidor, especialmente quando as atualizações são transmitidas através de um meio sem fio com recursos limitados (por exemplo, potência de transmissão e largura de banda da rede)
    - Ter mais clientes escalonados e carregar modelos locais simultaneamente pode resultar em grandes sobrecargas na comunicação, conexões mais instáveis e maior latência, o que pode levar a tarefas de aprendizagem com menos precisão
- Diante disso, os autores buscaram uma solução ideal que envolvesse o escalonamento do conjunto de clientes e o esquema de alocação de recursos em cada rodada, a fim de buscar o melhor desempenho de aprendizagem
- Contudo, esse problema de otimização é intratável devido à não-convexidade de sua função objetivo e por conta das restrições $(7a)$ e $(7d)$. Para resolver esse problema, os autores o dividiram em dois subproblemas:
    1. Determinar a política de escalonamento de clientes em cada rodada de comunicação
    2. Determinar uma alocação ótima de recursos para os clientes que foram selecionados

### Escalonamento de Clientes

- A ideia do escalonamento de clientes é limitar as trocas de comunicação, por meio da seleção de quais clientes irão transmitir seus modelos locais atualizados para o servidor
- Então, em vez de solicitar novos parâmetros de modelo local de todos os clientes, aplica-se a seguinte política de escalonamento:
- Durante cada rodada de comunicação $t$, o cliente com mensagens informativas está habilitado a realizar upload de seus novos parâmetros de modelo se um dos seguintes critérios de seleção for atendido:
    1. *[fórmula 8]* Essa condição compara o novo gradiente local com a cópia obsoleta no cliente: somente quando a diferença do gradiente for maior que as mudanças recentes em $w$, o novo modelo local será transmitido
    2. *[fórmula 9]* Além disso, para evitar clientes inativos por um longo período, força-se o upload dos parâmetros do modelo local o servidor se algum cliente $i$ não estiver ativo para transmitir novos parâmetros do modelo durante as últimas rodadas de comunicação. Para isso, define-se um relógio para cada cliente $i$, contando o número de rodadas de comunicação inativas desde a última vez que ele carregou seus modelos locais
- Se as condições não forem atendidas, o servidor reutilizará a cópia obsoleta do modelo local