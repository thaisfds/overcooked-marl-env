# Notas

python.exe -m pip install --upgrade pip

py -3.13 -m venv venv

venv\Scripts\activate

pip install gymnasium numpy pettingzoo # Adicione 'pettingzoo'

pip install -e .

python enviroment_tests/salad_lettuce.py > test_log.txt

deactivate

# Decisões
Resumo : ambiente gymnasium para MARL baseado em overcooked

### Design do Ambiente MARL (Multi-Agent)
##### Definir Agentes
Agentes Variáveis - Na implementação do step(), você precisará gerenciar um dicionário de agentes ativos e retornar os dados apenas para eles. O PettingZoo é a biblioteca ideal para isso, pois é construída para lidar com a natureza sequencial ou paralela de jogos com agentes variáveis/turnos.

##### Definir Observation Spaces
Observation Spaces (Visão Global) - Você usará um gymnasium.spaces.Box ou gymnasium.spaces.Dict para a observação. Recomendo um Box 3D (largura x altura x camadas de estado) onde cada camada representa o estado (e.g., 1ª camada: localização do agente, 2ª camada: localização do tomate, 3ª camada: estação de corte, etc.).

##### Definir Action Spaces
Action Spaces (Andar + Interagir) - Você usará um gymnasium.spaces.Discrete onde os primeiros 4 valores são as direções e os valores restantes são as interações (e.g., 0-3: Andar, 4: Interagir).

Colisão e Restrições - O desafio aqui é a Action Masking (mascaramento de ações). Em cada step(), o agente tentará uma ação, mas você precisará verificar se ela é válida (e.g., não há parede, estação ou agente no caminho). Se a ação for inválida, você pode: 
1) Não mover o agente (ficar parado), ou 
2) Aplicar uma pequena penalidade de recompensa.

##### Design da Recompensas
Recompensa Híbrida - Isso é excelente. No método step(), você calculará: 

Recompensa Individual (local): Se a ação foi uma interação de sucesso (e.g., pegou um item, colocou um item na estação).

Recompensa Coletiva (global): Se a ação resultou na entrega de um prato. Esta recompensa deve ser somada ao valor individual e distribuída para todos os agentes no dicionário rewards: dict.

##### Design da Conclusão
Tempo Limite - Isso se traduz no sinal truncated (truncamento). É a maneira do Gymnasium dizer que o episódio acabou por uma razão não diretamente relacionada à falha ou sucesso da tarefa (e.g., tempo esgotado).

Pratos Entregues - Isso se traduz no sinal terminated (terminação). É a conclusão natural da tarefa (sucesso ou falha crítica).

### Implementação do Env


# DICAS DA AI: Ordem de Implementação do Ambiente MARL

Fase 1: Esqueleto e Espaços (A Estrutura do Gymnasium/PettingZoo)

Esta fase é sobre definir a "forma" do seu ambiente. Se você errar aqui, nada mais funcionará.

    Criar a Classe Base: Defina a classe OvercookedMARL herdando de ParallelEnv (do PettingZoo).

    Definir Variáveis Estáticas: No __init__, defina a lista de IDs de agentes (self.agents) e o tamanho do mapa (self.map_size).

    Definir Observation & Action Spaces: Implemente self.observation_spaces e self.action_spaces como dicionários de gymnasium.spaces.

        Observação: Defina as dimensões exatas (largura x altura x camadas de estado) do seu Box 3D.

        Ação: Defina o número total de ações discretas (4 movimentos + 1 interagir) no seu Discrete(5).

    Implementar _initialize_game_state e _get_obs: Crie estas funções auxiliares. No início, elas podem retornar dados fictícios (np.zeros ou spaces.sample()) apenas para garantir que o formato de dados de entrada/saída esteja correto.

    Implementar reset(): Implemente o reset() para inicializar o estado do jogo e retornar as primeiras observações e informações usando seus métodos auxiliares.

Fase 2: Lógica Central (O Motor do Jogo)

Esta é a fase mais complexa, onde a simulação ocorre.

    Implementar step() (Lógica de Entrada): Faça o setup do método step(actions) para receber o dicionário de ações e inicializar os 5 dicionários de retorno (obs, rewards, term, trunc, info).

    Lógica de Movimento e Colisão: Dentro do loop do step(), implemente a lógica de movimentação dos agentes, garantindo que as regras de colisão (com paredes, estações e outros agentes) sejam respeitadas.

    Lógica de Interação: Implemente a lógica principal do Overcooked: o que acontece quando um agente usa a ação "Interagir" em diferentes estados (pegar item, cortar item, combinar item).

    Lógica de Recompensa Híbrida:

        Calcule as pequenas recompensas individuais (pela progressão).

        Calcule a grande recompensa global (pela entrega do prato) e distribua para todos os agentes.

    Lógica de Conclusão: Implemente as condições de terminated (pratos entregues) e truncated (tempo limite), garantindo que o dicionário de retorno __all__ seja marcado como True quando o episódio terminar.

Fase 3: Finalização e Testes

    Implementar render() (Opcional, mas Recomendado): Adicione a lógica para renderizar o mapa (ASCII simples, se estiver usando render_mode="human", ou uma matriz de pixels para render_mode="rgb_array").

    Executar check_env: Use a função check_env do Gymnasium (e parallel_api_test do PettingZoo, se desejar) para validar formalmente que você seguiu o protocolo da API.

    Rodar o Teste de Sanidade: Execute o script enviroment_tests/test.py para garantir que o ambiente roda 100 passos sem travar, confirmando que a lógica do jogo está estável.

    Finalizar o Setup: Garanta que seu setup.py e o registro no __init__.py estão corretos para que outros possam usar gym.make().

A Regra de Ouro: Não passe para a próxima fase até que o código da fase atual funcione perfeitamente com o seu código de teste.


