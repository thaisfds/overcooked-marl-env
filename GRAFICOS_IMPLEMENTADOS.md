# 📊 Gráficos Implementados - Avaliação de Modelos

## Resumo das Implementações

Este documento descreve todas as visualizações implementadas no notebook `Avaliacao_Modelos.ipynb`.

---

## 📍 Seção 4: Análise Comparativa de Performance (Training)

### 4.1 - Curvas de Aprendizado (Top 5 Modelos)
**Arquivo:** `midia/learning_curves_top5.png`

**Características:**
- Mostra a evolução do reward durante o treinamento dos 5 melhores modelos
- Suavização com média móvel para reduzir ruído
- Bandas de variação (desvio padrão) para os top 3 modelos
- Cores distintas para cada modelo
- Legenda com ranking e reward final

**Insights:**
- Identifica convergência e estabilidade de treinamento
- Detecta overfitting (curva que sobe e depois cai)
- Compara velocidade de aprendizado entre configurações

---

### 4.2 - Ranking e Distribuição de Performance

**Arquivo:** `midia/ranking_distribuicao.png`

#### 4.2.1: Ranking Horizontal
- Barras horizontais coloridas por gradiente de performance
- Top 3 destacados: 🥇 Ouro, 🥈 Prata, 🥉 Bronze
- Valores exibidos nas barras
- Ordenação por reward decrescente

#### 4.2.2: Box Plot com Distribuição
- Box plot mostrando quartis (Q1, Mediana, Q3)
- Pontos individuais sobrepostos
- Linha de média destacada
- Anotações com valores dos quartis

**Insights:**
- Visualiza spread e outliers
- Identifica consistência geral
- Mostra assimetria da distribuição

---

### 4.3 - Análise de Hiperparâmetros

**Arquivo:** `midia/hiperparametros_analise.png`

#### 4.3.1: Scatter 3D-like (Learning Rate vs Update Epochs)
- Cores representam reward
- Tamanho das bolhas representa num_steps
- Escala logarítmica no eixo X
- Melhor modelo marcado com estrela dourada

#### 4.3.2: Heatmap de Correlação
- Correlação entre todos os hiperparâmetros e reward
- Escala de cores coolwarm (-1 a +1)
- Valores numéricos anotados
- Identifica relações positivas/negativas

**Insights:**
- Descobre quais hiperparâmetros impactam mais o reward
- Identifica interações entre parâmetros
- Guia otimizações futuras

---

### 4.4 - Estatísticas Resumidas

**Formato:** Tabela impressa no console

**Métricas incluídas:**
- Melhor/Pior reward
- Média e mediana
- Desvio padrão
- Quartis (Q1, Q3)
- Amplitude e IQR
- Coeficiente de Variação (CV%)

---

## 📍 Seção 5: Análise Avançada de Avaliação

### 5.1 - Gráficos Avançados de Avaliação

**Arquivo:** `midia/analise_avancada_avaliacao.png`

#### 5.1.1: Training vs Evaluation (Scatter)
- Scatter plot com linha de identidade (y=x)
- Cores representam success rate
- Barras de erro para variância
- Modelos anotados
- Detecta overfitting (pontos abaixo da linha)

#### 5.1.2: Violin Plot - Distribuição de Rewards
- Mostra distribuição completa de rewards para cada modelo
- Mediana e média marcadas
- Cores gradientes por modelo
- Revela multimodalidade e assimetria

#### 5.1.3: Scatter - Eficiência (Reward/Steps)
- Eixo X: Episode Length
- Eixo Y: Evaluation Reward
- Tamanho: Success Rate
- Cores: Eficiência (reward/steps)
- Linha de tendência
- Identifica modelos eficientes (alto reward, poucos steps)

#### 5.1.4: Overfitting Score (Train - Eval)
- Barras horizontais coloridas
- Verde: sem overfitting (< 5)
- Laranja: overfitting leve (5-10)
- Vermelho: overfitting alto (> 10)
- Linhas de referência marcadas

#### 5.1.5: Radar Chart - Top 3 Modelos
- Comparação multidimensional dos 3 melhores
- 5 métricas normalizadas:
  - Eval Reward
  - Success Rate
  - Efficiency
  - Stability (1 - std)
  - Training Reward
- Cores: 🥇 Ouro, 🥈 Prata, 🥉 Bronze
- Área preenchida

#### 5.1.6: Tabela de Métricas Detalhadas
- Tabela completa de todas as métricas
- Linhas alternadas para legibilidade
- Header colorido
- Inclui: Train R, Eval R±σ, Length, Success %, Efficiency, Overfitting

**Métricas Consolidadas (Console):**
- 🏆 Melhor Avaliação
- 📊 Mais Estável (menor σ)
- ⚡ Mais Eficiente
- ✅ Menos Overfitting
- ⚖️ Modelo Mais Balanceado (distância euclidiana ao ideal)

---

## 📍 Seção 6: Visualização do Comportamento dos Modelos

### 6.1 - Comparação Visual: Melhor vs Pior Modelo

**Arquivo:** `midia/comparacao_melhor_vs_pior.png`

**Layout:**
- **Linha Superior:** 5 frames do MELHOR modelo (borda verde)
- **Linha Média:** 5 frames do PIOR modelo (borda vermelha)
- **Linha Inferior:** 3 gráficos comparativos

#### Gráficos Comparativos:

1. **Comparação de Rewards**
   - Barras agrupadas: Training vs Evaluation
   - Cores: Verde (melhor), Vermelho (pior)

2. **Métricas de Performance**
   - Steps e Success Rate lado a lado
   - Comparação direta

3. **Ganhos do Melhor Modelo**
   - Barras horizontais de delta percentual
   - Reward Gain %
   - Steps Saved %
   - Success Δ %
   - Valores anotados

**Análise Textual (Console):**
- 📊 Diferença de Performance numérica
- 💡 Insights automáticos baseados nos deltas
- Classificação da melhoria (substancial/significativa/moderada)

---

## 📍 Seção 7: Relatório Final

### Arquivo JSON gerado

**Arquivo:** `midia/relatorio_avaliacao_modelos.json`

**Conteúdo:**
- Configuração de cada modelo avaliado
- Métricas de training e evaluation
- Ranking completo
- Timestamp da avaliação

---

## 🎨 Paleta de Cores Utilizada

### Por Tipo de Gráfico:

- **Rankings:** Gradiente RdYlGn (Vermelho → Amarelo → Verde)
- **Top 3:** Ouro (#FFD700), Prata (#C0C0C0), Bronze (#CD7F32)
- **Heatmaps:** Coolwarm (Azul ↔ Vermelho)
- **Scatter Rewards:** Viridis, Plasma
- **Overfitting:** Verde (ok), Laranja (leve), Vermelho (alto)
- **Comparações:** Verde (melhor), Vermelho (pior)

---

## 📈 Métricas Calculadas

### Métricas Principais:
1. **Training Reward:** Média dos últimos 100 episódios do treinamento
2. **Evaluation Reward:** Média de 20 episódios de avaliação
3. **Success Rate:** % de episódios com reward > -50
4. **Episode Length:** Número médio de steps por episódio
5. **Stability:** Desvio padrão dos rewards de avaliação

### Métricas Derivadas:
6. **Efficiency:** Reward / Steps (quanto reward por step)
7. **Overfitting Score:** Training Reward - Evaluation Reward
8. **Balanced Score:** Distância euclidiana ao ponto ideal (1,1,1,1)

---

## 🎯 Como Interpretar

### Sinais de Bom Modelo:
- ✅ Evaluation reward próximo ou maior que training reward
- ✅ Baixo desvio padrão (estável)
- ✅ Alta success rate (> 70%)
- ✅ Curva de aprendizado suave e convergente
- ✅ Alta eficiência (reward/steps)
- ✅ Overfitting score < 5

### Sinais de Problema:
- ⚠️ Gap grande entre training e evaluation (overfitting)
- ⚠️ Alto desvio padrão (instável)
- ⚠️ Baixa success rate
- ⚠️ Curva de aprendizado errática
- ⚠️ Muitos steps para pouco reward

---

## 🔧 Customizações Possíveis

Para ajustar as visualizações:

```python
# Número de modelos nas curvas de aprendizado
top_n = min(5, len(config_models))  # Altere 5 para outro valor

# Janela de suavização
window = min(100, len(episode_rewards) // 10)  # Altere 100

# Threshold de overfitting
# Linha 1: 5 (leve)
# Linha 2: 10 (alto)

# Número de episódios de avaliação
num_episodes=20  # Na função load_and_evaluate_model

# Frames na comparação visual
num_episodes=1  # visualize_model_behavior para comparação
```

---

## 📦 Arquivos Gerados

Todos os gráficos são salvos em `midia/`:

1. `learning_curves_top5.png` - Curvas de aprendizado
2. `ranking_distribuicao.png` - Ranking e box plot
3. `hiperparametros_analise.png` - Scatter e heatmap
4. `analise_avancada_avaliacao.png` - 6 gráficos de avaliação
5. `comparacao_melhor_vs_pior.png` - Comparação visual
6. `visualizacao_melhor_modelo.png` - Comportamento do melhor
7. `relatorio_avaliacao_modelos.json` - Dados brutos

---

## 🚀 Próximos Passos Sugeridos

1. **Testes A/B:** Comparar modelos em cenários específicos
2. **Análise Temporal:** Como performance muda ao longo do treinamento
3. **Ablation Studies:** Remover features e ver impacto
4. **Ensemble Methods:** Combinar melhores modelos
5. **Transfer Learning:** Usar melhor modelo como ponto de partida

---

**Implementado em:** 17/11/2025  
**Notebook:** `Avaliacao_Modelos.ipynb`  
**Versão:** 2.0 - Análise Avançada Completa
