# Plano de Ação - Customização e Testes de Gameplay (NewOvercookedEnv)

Este plano de ação apresenta as melhorias implementadas no ambiente de aprendizado por reforço para facilitar a criação de novos mapas, novas receitas e ingredientes personalizados, além de habilitar testes interativos de gameplay diretamente via teclado.

---

## 🛠️ O que foi Alterado (Alinhado com SOLID)

1. **Responsabilidade Única (SRP):**
   * A lógica de testes manuais foi totalmente desacoplada no script `example/keyboard_play.py`.
   * A carga e registro dinâmico de novos ingredientes e receitas foi centralizada na classe `RecipeManager`.
2. **Aberto para Extensão, Fechado para Modificação (OCP):**
   * Novos mapas, ingredientes (com cores e tempos de corte customizados) e receitas podem ser adicionados dinamicamente sem necessidade de alterar o código interno da simulação física ou do construtor de observações RL.
3. **Validação Robusta:**
   * A classe `MapLoader` agora possui um ponto de entrada CLI para testar e validar novos mapas e receitas de forma rápida e segura.

---

## 🗺️ 1. Como Criar Novos Mapas

Os mapas são definidos em arquivos de texto plano (`.txt`), estruturados em três seções principais demarcadas por cabeçalhos de comentário.

### Estrutura de Caracteres do Grid:
* `_` e `|` : Paredes externas.
* `#` : Balcões/estantes padrão (bloqueiam passagem, permitem pousar itens).
* Espaço em branco ` ` : Área transitável para os agentes.
* `0`, `1` : Posições iniciais dos agentes.
* `C` : Tábua de corte (Chopping Board).
* `P` : Dispenser de pratos (Plate Dispenser).
* `D` : Área de entrega de pratos (Delivery).
* `T` : Lixeira (Trash).
* `I` : Posição genérica de dispenser de ingrediente (mapeado dinamicamente na seção `# Dispensers`).

### Exemplo Completo de Mapa (`meu_mapa.txt`):
```text
# Grid Map
_________
|#C#P#D#|
|# 0   #|
|#   1 #|
|#I#I#I#|
_________

# Dispensers (name x y)
tomato 1 4
lettuce 3 4
onion 5 4

# Recipes
SIMPLE_SALAD, ONION_SOUP, ONION_SALAD
```

---

## 🥗 2. Como Adicionar Novas Receitas e Ingredientes (`recipes.json`)

Para evitar criar novas subclasses em Python e compilar o código toda vez que uma receita ou ingrediente for adicionado, você pode colocar um arquivo `recipes.json` **na mesma pasta** que o seu arquivo de mapa. O ambiente carregará esse arquivo de forma totalmente transparente antes de validar e iniciar o jogo.

### Estrutura do `recipes.json`:
* `ingredients` (Opcional): Lista de ingredientes customizados. Cada ingrediente define:
  * `name`: Nome do ingrediente (case-insensitive).
  * `color`: Cor em RGB `[R, G, B]` usada na renderização.
  * `cuts_needed` (Padrão: 3): Quantas interações de corte são necessárias na tábua `C`.
  * `char`: Letra que representa o ingrediente em formato ASCII.
* `recipes`: Lista de receitas. Cada receita define:
  * `name`: Nome identificador (deve bater com a seção `# Recipes` do mapa).
  * `ingredients`: Lista contendo os ingredientes exatos requeridos.
  * `reward`: Recompensa (score) obtida na entrega bem-sucedida.

### Exemplo de `recipes.json`:
```json
{
  "ingredients": [
    {
      "name": "onion",
      "color": [235, 150, 235],
      "cuts_needed": 3,
      "char": "o"
    }
  ],
  "recipes": [
    {
      "name": "ONION_SOUP",
      "ingredients": ["onion", "onion", "onion"],
      "reward": 180.0
    },
    {
      "name": "ONION_SALAD",
      "ingredients": ["lettuce", "onion"],
      "reward": 130.0
    }
  ]
}
```

---

## 🔍 3. Como Validar Mapas via CLI

Antes de treinar agentes RL ou rodar gameplays, você pode validar se a estrutura do grid, a posição dos dispensadores e as receitas estão consistentes.

Execute o comando:
```bash
python -m overcooked_marl_env.map.map_loader caminho/do/seu_mapa.txt
```

> [!TIP]
> Se houver um arquivo `recipes.json` no mesmo diretório do arquivo do mapa, o validador irá carregá-lo automaticamente e validar suas receitas customizadas sem erros.

---

## 🎮 4. Como Testar Gameplay via Teclado

Implementamos um runner interativo baseado em Pygame que permite a humanos controlarem os agentes para inspecionar física, colisões e mecânicas de cozinha.

Para iniciar o teste com o mapa padrão/template:
```bash
python example/keyboard_play.py
```

Para carregar um mapa específico:
```bash
python example/keyboard_play.py --map caminho/do/seu_mapa.txt
```

### Controles do Jogo:
* **Agente 0 (Azul):**
  * Movimento: `W` (Cima), `A` (Esquerda), `S` (Baixo), `D` (Direita)
  * Interação: Barra de Espaço (`Spacebar`) (pegar, cortar, colocar, entregar)
* **Agente 1 (Laranja):**
  * Movimento: Teclas de Direção (Setas do Teclado)
  * Interação: Tecla `Enter` / `Return`
* **Gerais:**
  * Reiniciar mapa: Tecla `R`
  * Sair do jogo: Tecla `Esc` ou fechar a janela.

---

## 🧪 5. Execução dos Testes Automatizados

Após qualquer modificação, garanta a integridade da suíte de testes existente rodando:
```bash
python -m unittest discover -s overcooked_marl_env/tests
```
E valide a cooperação paralela dos agentes rodando o script de exemplo padrão:
```bash
python example/default_map_test.py
```
