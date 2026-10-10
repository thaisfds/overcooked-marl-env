# Análise de Desempenho dos Modelos e Engenharia de Recompensas (PPO 150D)

* **Data da Análise:** 09/10/2026
* **Ambiente:** `overcooked_marl_env` (Espaço de Observação Featurizado 150D)
* **Sessão de Experimentos:** `PPO_Marl_20261009_001548` (24 Trials de Grid Search)
* **Status do Repositório:** Mudanças restauradas via `git stash pop`

---

## 1. Contexto e Resumo Executivo

Após a implantação do novo espaço observacional de **150 dimensões** (`featurized_observation_builder.py`), 24 combinações de hiperparâmetros foram treinadas por 1.000 episódios. Uma avaliação qualitativa e quantitativa foi realizada através de:
1. Métricas de treino no Ray/Tune (`latest_summary.md`);
2. Avaliação estática de 20 episódios (`evaluation_results.json`);
3. Rollouts visuais com telemetria detalhada de 20s (400 steps) para todos os 24 modelos (`visualization_summary.md` e `models_20s/*.log`).

### Principal Diagnóstico:
O **Modelo 9 (`940b6_00009`)** destacou-se como o único modelo capaz de avançar na cadeia de produção de **receitas complexas de 3 ingredientes (`FRESH_SALAD`)**, interagindo ativamente com Alface, Tomate e Pepino. No entanto, o agente acumulou ingredientes e pratos nas bancadas sem concluir as entregas compostas.

A investigação comprova que **a causa não é falta de entropia (exploração)**, mas sim uma **armadilha de saturação de recompensas intermediárias (*Reward Shaping Trap*)** somada à ausência de incentivo direcional para condução do prato final até a estação de entrega.

---

## 2. Diagnóstico Detalhado do Modelo 9 (`940b6_00009`)

### 2.1. Hiperparâmetros e Resultados Globais
* **Trial ID:** `PPO_overcooked_marl_parallel_940b6_00009_9`
* **Learning Rate (LR):** `0.0003`
* **Fator de Desconto ($\gamma$):** `0.98`
* **Coeficiente de Entropia:** `0.03` *(o maior valor avaliado no grid search)*
* **Batch Size de Treino:** `4000`
* **Recompensa Média no Treino:** **+5.765,87** *(Rank #2 geral entre os 24 modelos)*
* **Recompensa no Rollout 20s:** **+207.94** *(Rank #2 geral)*
* **Entregas Concluídas (20s):** 1 (`SIMPLE_SALAD`)
* **Ingredientes Manipulados:** Alface (28 steps), Tomate (142 steps), Pepino (21 steps)

---

### 2.2. Anatomia da Trajetória Passo a Passo (Rollout de 20s / 400 Steps)

```
[Step 0001 - 0176]  Alface -> Tábua -> 3 Cortes -> Prato -> Entrega (SIMPLE_SALAD: +110.0 pts)
[Step 0177 - 0185]  Pega Pepino (Cucumber) -> Move para a bancada central
[Step 0220 - 0242]  Pega Tomate (Tomato) -> Leva à Tábua -> 3 Cortes completos (+9.0 pts)
[Step 0324 - 0391]  Pega Prato Vazio (+3.0 pts) -> Monta Prato com Alface na bancada (+13.54 pts)
[Step 0388 - 0400]  Circula com prato incompleto na entrega (-2.0 pts) -> Tempo esgotado (400 steps)
```

1. **Fase 1 — Entrega Rápida da Receita de 1 Ingrediente (Steps 1–176):**
   - O agente coletou alface no dispenser `(3, 5)`, levou à tábua em `(3, 1)`, executou os 3 cortes necessários (ganhando $+1.0, +1.0$ e milestone de $+8.0$), colocou no prato (`Plate (lettuce)`: $+32.0$ pts) e caminhou até a entrega em `(5, 3)`.
   - **Step 176:** Entrega de `SIMPLE_SALAD` concluída com sucesso (+110.0 pontos no step).
2. **Fase 2 — Exploração Ativa dos 3 Ingredientes (Steps 177–242):**
   - Com `SIMPLE_SALAD` entregue, as ordens restantes na fila HUD passaram a ser:
     - `GARDEN_SALAD` (2 ingredientes: alface + tomate);
     - `FRESH_SALAD` (3 ingredientes: alface + tomate + pepino).
   - O modelo reagiu prontamente à nova demanda:
     - No **step 181**, pegou `Cucumber (raw)` no dispenser `(4, 5)`.
     - No **step 220**, pegou `Tomato (raw)` no dispenser `(2, 5)`.
     - No **step 234**, posicionou o tomate na tábua de corte.
     - Nos **steps 235 a 242**, realizou os cortes até a conclusão (`Tomato (ready)`), faturando o milestone de $+9.0$ pontos.
3. **Fase 3 — O Gargalo da Montagem e Entrega Composta (Steps 250–400):**
   - O agente retirou outro prato no dispenser (step 324) e preparou montagens parciais em bancadas (step 391: `Plate (lettuce)` com $+13.54$ pts).
   - No entanto, devido à ausência de tábuas adicionais e de um vetor de recompensa para reunir os 3 itens picados no mesmo prato, o agente ficou circulando entre as bancadas e a estação de entrega segurando prato vazio ou com alface residual, sofrendo penalidades de entrega inválida ($-2.0$ nos steps 388 e 389) até o encerramento do episódio.

---

## 3. Mapeamento das Recompensas Atuais

### 3.1. Valores Definidos em `overcooked_marl_env/mechanics/reward_shaper.py`

| Categoria | Parâmetro | Valor Atual | Descrição / Efeito no Agente |
| :--- | :--- | :--- | :--- |
| **Custo de Vida** | `step_cost` | `-0.005` | Custo por step para desencorajar ociosidade |
| **Dispenser** | `progress_pickup_needed` | `+5.0` | Coleta de ingrediente requisitado |
| **Dispenser** | `unneeded_pickup_penalty` | `0.0` | Sem penalidade por coletar ingrediente não usado |
| **Tábua de Corte** | `place_on_chop` | `+5.0` | Posicionar ingrediente cru na tábua |
| **Tábua de Corte** | `progress_place_on_chop` | `+5.0` | Bônus de progresso por colocar na tábua |
| **Processamento** | `chop_cut` | `+1.0` | Recompensa por cada corte bem-sucedido |
| **Processamento** | `chop_completed` | `+8.0` | Milestone ao concluir o corte (ex: 3º corte) |
| **Coleta Processado** | `progress_pickup_ready` | `+2.0` | Retirar item picado da tábua |
| **Montagem** | `progress_combine_ready` | `+10.0` | Juntar ingrediente picado com o prato |
| **Montagem** | `progress_ingredient_added_base` | `+2.0` | Base por adicionar ingrediente ao prato |
| **Montagem** | `progress_ingredient_added_increment` | `+2.0` | Incremento por ingrediente já presente |
| **Montagem** | `plate_complete_bonus` | `+15.0` | Prato montado com todos os ingredientes da receita |
| **Montagem** | `progress_pickup_complete_plate` | `+5.0` | Pegar prato totalmente finalizado |
| **Entrega Base** | `recipe_reward` | Ver 3.2 | Recompensa intrínseca da receita |
| **Entrega Bônus** | `order_bonus_mult` | `0.5` | Bônus se entregue na ordem exata do HUD (FIFO) |
| **Multiplicador** | `delivery_multiplier_step` | `+0.5` | Escala: $1.0\times$ (1ª), $1.5\times$ (2ª), $2.0\times$ (3ª entrega) |
| **Vitória** | `episode_complete_bonus` | `+5000.0` | Bônus massivo ao zerar todas as ordens |
| **Penalidades** | `invalid_delivery_penalty` | `-2.0` | Tentar entregar item ou prato errado |
| **Penalidades** | `useless_cycle_penalty` | `-5.0` | Picar ou soltar ciclicamente sem sentido |

---

### 3.2. Valores das Receitas em `overcooked_marl_env/core/default_recipes.json`

| Receita | Ingredientes Requeridos | Recompensa Base de Entrega | Esforço (Cortes / Deslocamentos) |
| :--- | :--- | :--- | :--- |
| `SIMPLE_SALAD` | `["lettuce"]` (1) | **20.0** | 1 coleta + 3 cortes + 1 prato |
| `GARDEN_SALAD` | `["lettuce", "tomato"]` (2) | **25.0** | 2 coletas + 6 cortes + 1 prato |
| `FRESH_SALAD` | `["lettuce", "tomato", "cucumber"]` (3) | **30.0** | 3 coletas + 9 cortes + 1 prato |

---

## 4. Análise de Causa Raiz: Por que o Modelo não Entregou?

### 4.1. Desalinhamento da Recompensa Marginal (*Reward Shaping Trap*)
Ao comparar a recompensa de **preparação intermediária** com o ganho de **entrega final**:
* Preparar **1 ingrediente extra** (pegar $+5$, pôr na tábua $+10$, picar $+11$, prato $+12$) concede cerca de **$+38.0$ pontos em shaping**.
* Para 2 ingredientes extras (transformar `SIMPLE_SALAD` em `FRESH_SALAD`), o agente acumula **~+76.0 pontos** só de manipulação nas bancadas.
* No entanto, a recompensa base de entrega ao passar de `SIMPLE_SALAD` para `FRESH_SALAD` aumenta de $20.0$ para $30.0$ — uma mísera diferença de **apenas $+10.0$ pontos**!

> **Conclusão:** O gradiente do RL descobre que o caminho de menor risco e maior recompensa cumulativa é **picar ingredientes e montar pratos nas bancadas**, evitando o risco da penalidade de $-2.0$ na entrega.

---

### 4.2. Entropia vs Recompensas: Qual deve ser ajustado?

| Hipótese | Avaliação Técnica | Decisão |
| :--- | :--- | :--- |
| **Aumentar Entropia?** | O Modelo 9 já utilizou `entropy_coeff = 0.03`, o maior do grid search. Modelos com 0.01 nem sequer tocaram no tomate. Aumentar além de 0.03 tornará a política excessivamente estocástica ("ruidosa"), quebrando trajetórias coordenadas longas (como transportar o prato pronto até a entrega sem errar o caminho). | **NÃO aumentar a entropia.** Manter `0.03`. |
| **Ajustar Recompensas?** | A razão entre recompensas intermediárias e entrega final está desbalanceada. Não há incentivo de potencial por distância (potential-based shaping) para orientar o agente quando ele segura um prato válido até a estação de entrega. | **SIM, ajustar as recompensas.** |

---

### 4.3. Gargalos Estruturais do Mapa (`default_map.txt`)
1. **Apenas 1 Tábua de Corte (`B` em `3, 1`):**
   - Para preparar 3 ingredientes, o agente precisa cortar o ingrediente 1, removê-lo da tábua, colocá-lo em uma bancada livre, buscar o ingrediente 2, e repetir o ciclo. Sem uma heurística de desobstrução, a bancada acumula itens e o fluxo trava.
2. **Dispenser Não Utilizado de Cebola (`O` em `5, 5`):**
   - O mapa contém um dispenser de cebola, mas nenhuma receita do HUD utiliza cebola. Modelos com entropia gastaram dezenas de steps picando cebola inutilmente.

---

## 5. Plano de Ação Recomendado para os Próximos Treinos

1. **Rebalanceamento das Recompensas de Receitas:**
   - Atualizar `default_recipes.json` para refletir o esforço real:
     - `SIMPLE_SALAD` (1 item): **20.0**
     - `GARDEN_SALAD` (2 itens): **60.0** *(em vez de 25.0)*
     - `FRESH_SALAD` (3 itens): **120.0** *(em vez de 30.0)*
2. **Shaping Direcional para a Estação de Entrega:**
   - Adicionar uma recompensa baseada em potencial ($\Phi(s) = -\text{dist}(pos, delivery)$) quando o agente estiver segurando um prato montado com uma receita pendente válida, guiando-o suavemente até `(5, 3)` ou `(6, 3)`.
3. **Penalidade de Estagnação / Descarte de Pratos nas Bancadas:**
   - Penalizar suavemente o abandono de pratos prontos em bancadas que não avançam para a entrega.
4. **Fixação dos Melhores Hiperparâmetros:**
   - Basear os próximos treinos nos hiperparâmetros campeões do Modelo 9:
     - `learning_rate: 0.0003`
     - `gamma: 0.98`
     - `entropy_coeff: 0.03`
     - `train_batch_size: 4000`

---

## 6. Alterações Implementadas no Código

* **Data da Implementação:** 09/10/2026
* **Arquivos Modificados:**
  1. [`overcooked_marl_env/core/default_recipes.json`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/core/default_recipes.json): Recompensas base atualizadas para `GARDEN_SALAD` (60.0) e `FRESH_SALAD` (120.0). Definido como **Single Source of Truth** estrito.
  2. [`overcooked_marl_env/core/recipes.py`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/core/recipes.py): Código legado totalmente eliminado (removidas classes hardcoded `SimpleSaladRecipe`, `GardenSaladRecipe`, `FreshSaladRecipe`). O `RecipeManager` e o dicionário `RECIPE_DEFINITIONS` agora são carregados e populados dinamicamente estritamente a partir do `default_recipes.json`.
  3. [`overcooked_marl_env/mechanics/reward_shaper.py`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/mechanics/reward_shaper.py):
     - Adicionado `delivery_guidance_weight: float = 1.0` em `RewardValues`.
     - Implementado o método `calculate_step_shaping` com *potential-based reward shaping* (Ng et al., 1999): bonifica $+1.0$ por passo de aproximação à entrega quando o agente segura prato com receita válida pendente, e $-1.0$ ao afastar-se (balanço líquido de ida e volta estritamente nulo, à prova de farming).
  4. [`overcooked_marl_env/mechanics/interaction_handler.py`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/mechanics/interaction_handler.py): Delegado `calculate_step_shaping` para `RewardShaper`.
  5. [`overcooked_marl_env/env.py`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/env.py): No `step()`, após a resolução de movimento (Step 4), aplica `calculate_step_shaping` diretamente nas recompensas do agente.
  6. [`overcooked_marl_env/tests/unit/test_extensibility_architecture.py`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/tests/unit/test_extensibility_architecture.py): Adicionados testes unitários validando as novas recompensas de receitas e a conservação de potencial do `calculate_step_shaping`.

---

## 7. Diagnóstico Empírico das Telemetrias de 400 Steps (Fase 2) e Overhaul Arquitetural

* **Data da Análise e Resolução:** 10/10/2026
* **Arquivos Analisados:**
  - `experiments/logs/vision_analysis/visualization_summary.md`
  - `experiments/logs/vision_analysis/models_20s/model_0_trial_8ddbd_00000.log`
  - `experiments/logs/vision_analysis/models_20s/model_1_trial_8ddbd_00001.log`
  - `experiments/logs/vision_analysis/models_20s/model_2_trial_8ddbd_00002.log`
  - `experiments/logs/latest_summary.md`

### 7.1. Patologias Empíricas Identificadas
1. **The Counter Farming Trap (Farming Intermediário de Pratos):**
   - No trial `8ddbd_00000` (entropy 0.02), o agente acumulou +675 pts nos primeiros 400 steps, mas apenas **1 prato** (`SIMPLE_SALAD`) foi entregue.
   - O agente montou 5 pratos com alface (+58 a +62 pts por montagem) e abandonou 4 deles em balcões livres `(5,2)`, `(5,4)` e `(2,2)` sem jamais levá-los à entrega, explorando a recompensa imediata de combinação sem concluir o ciclo.
2. **Ponto Cego e Abandono Pós-Montagem para Receitas Complexas:**
   - Para a 2ª receita (`GARDEN_SALAD` = Alface + Tomate), o agente picou tomate e completou o prato no balcão `(5,4)` (step 191). No step 198, pegou o prato pronto, mas no step 199 largou-o de volta na bancada sem qualquer penalidade.
   - Mais tarde (step 255), ao tentar esvaziar a cozinha, pegou um prato antigo de alface abandonado e entregou-o, sofrendo penalidade de `-20.0` por receita não solicitada (`unrequested_delivery`), entrando em loop defensivo de congelamento.
3. **Colapso com Alta Entropia:**
   - Trial `8ddbd_00002` (`entropy_coeff = 0.04`) colapsou para +35.26 pts com 0 entregas, comprovando que `0.04` destrói trajetórias longas coordenadas.

---

### 7.2. Melhorias Implementadas no Código

1. **Rebalanceamento de Recompensas e Anti-Exploitation:**
   - [`RewardValues` em `reward_shaper.py`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/mechanics/reward_shaper.py):
     - Reduzidas recompensas intermediárias exploráveis: `progress_pickup_needed: 3.0`, `place_on_chop: 3.0`, `chop_cut: 0.5`, `chop_completed: 5.0`, `progress_combine_ready: 8.0`.
     - Incrementadas recompensas de finalização: `plate_complete_bonus: 20.0`, `progress_pickup_complete_plate: 10.0`.
     - Adicionadas penalidades contra desperdício e farming: `abandon_dish_penalty: -10.0`, `unneeded_pickup_penalty: -2.0`, `counter_clutter_penalty: -1.0`, `unrequested_delivery_penalty: -10.0`.
   - Prevenção de duplicação: `_check_ingredient_already_processed` agora ignora o próprio item recém-pego e impede que o agente retire ingredientes repetidos dos dispensers quando já existem instâncias suficientes na cozinha.

2. **Potential-Based Shaping Bidirecional ($\Phi(s)$):**
   - Agente segurando prato pronto com receita pendente: guiado em direção à estação de entrega ($\Delta \Phi = \text{dist}_{old} - \text{dist}_{new}$).
   - Agente com as mãos vazias e prato pronto esquecido em bancada: guiado em direção ao balcão com o prato pronto para resgatá-lo e entregá-lo.

3. **Mecânica de Bancadas e Estações:**
   - [`CounterHandler`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/mechanics/interactions/counter_handler.py): Aplica `abandon_dish_penalty` (-10.0) se o agente depositar um prato pronto válido em uma bancada em vez de entregá-lo. Concede `progress_pickup_complete_plate` (+10.0) ao resgatá-lo.
   - [`DeliveryHandler`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/mechanics/interactions/delivery_handler.py): Penalidade calibrada para `-10.0` uniforme tanto no retorno local quanto global para entregas não solicitadas.
   - [`DispenserHandler`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/mechanics/interactions/dispenser_handler.py): Cálculo de recompensa de progresso desacoplado da instância em mãos para garantir consistência perfeita no lifecycle de preparo.

4. **Hierarquia Observacional no Vetor 150D:**
   - [`FeaturizedObservationBuilder`](file:///home/thaiswsl/NewOvercookedEnv/overcooked_marl_env/environment/featurized_observation_builder.py):
     - **Bloco 1 (Self State):** `is_holding_target` estendido para qualquer receita pendente no HUD (1.0 para `rec_1`, 0.8 para `rec_2`/`rec_3`). `is_holding_invalid = 1.0` se segurar receita já entregue ou incorreta.
     - **Bloco 5 (Attentive Dispensers):** Dispensers observam pratos parciais nos balcões e orientam o agente a buscar apenas os ingredientes faltantes.
     - **Bloco 6 (Appliance Beacon):** Arquitetura hierárquica em 5 níveis (prato pronto $\to$ entrega; ingrediente cru $\to$ tábua; ingrediente pronto $\to$ prato no balcão/dispenser de prato; prato incompleto $\to$ ingrediente pronto; mãos vazias $\to$ prato pronto no balcão / ingrediente na tábua / dispenser necessário). Dimensões mantidas estritamente no vetor 150D.

5. **Ajuste de Hiperparâmetros PPO:**
   - [`train_ppo.py`](file:///home/thaiswsl/NewOvercookedEnv/experiments/training/train_ppo.py): Espaço de busca de `entropy_coeff` ajustado para `[0.02, 0.025, 0.03]`, eliminando a instabilidade observada com 0.04.

---

### 7.3. Validação com Testes Unitários
* **Comando executado:** `venv/bin/python -m unittest discover -s overcooked_marl_env/tests`
* **Resultado:** **196/196 testes aprovados com sucesso (0 falhas, 0 erros em 0.154s).**

