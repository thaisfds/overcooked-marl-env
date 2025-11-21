Os modelos não funcionam com recompensa negativa. Eles param de explorar o ambiente e de interagir com os ingredientes disponiveis

O uso de recompensas baixas fez o modelo achar uma forma de trapacer, ganhando ponto sempre pegando e colocando itens nas bancadas (coloca e tira infinitamente)

======================================================================
TRAINING COMPLETED
======================================================================
Total time: 73.2 min (1.22h)
Configurations tested: 72
Best reward: 1123.25 ± 700.85
Models saved in: models

modifique o model evaluation para que:
- cada rodada de experimentos fique salvo em uma pasta com o timestampe da run
- Gere um grafico com as receitas entregues, e em qual ordem se possivel pelos melhores modelos (quantos, quais, e em qual ordem)
- Remova o grafico Success Rate (Episodes with Deliveries), Delivery Distribution - Top Models
- Melhore a descrição de  Detailed Comparison: Top 4 by Success Rate vs Top 3 by Reward, trocando esse S#1, S#2 , S#3 para first, second, ....
- Melhore o Comparative Table: Top 5 Models by Success Rate vs Reward, Tornando a escrita no rank melhor ( use, first, second, ....). E adicione novas informações uteis. Também aumente a fonte utilizada
- Salve todos os dados de EVALUATION SUMMARY em um arquivo, e salve para todos os modelos em ordem de success rate e depois em caso de empate a Reward

======================================================================
TREINAMENTO CONCLUÍDO (IDPPO com CNN)
Total time: 158.9 min
Modelos salvos em: models
======================================================================