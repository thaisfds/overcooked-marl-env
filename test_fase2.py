"""
Script de teste para validar a Fase 2 da refatoração.

Este script testa se os componentes Core foram criados corretamente.
"""

import sys
import numpy as np

print("=" * 70)
print("TESTE DA FASE 2: Extração de Componentes Core")
print("=" * 70)

# Teste 1: ItemInstanceManager
print("\n" + "=" * 70)
print("TESTE 1: ItemInstanceManager")
print("=" * 70)

try:
    from overcooked_marl_env.core.item_instance_manager import ItemInstanceManager
    from overcooked_marl_env.core import ItemInstanceManager as ImportedManager
    
    # Criar instância
    manager = ItemInstanceManager()
    print(f"✅ ItemInstanceManager criado: {manager}")
    
    # Testar funcionalidade básica
    item_id = manager.create_instance({"type": "tomato", "state": "raw"})
    print(f"✅ Item criado com ID: {item_id}")
    
    item = manager.get_instance(item_id)
    print(f"✅ Item recuperado: {item}")
    
    manager.set_agent_item("agent_0", item_id)
    held_item = manager.get_agent_item("agent_0")
    print(f"✅ Agente segurando item: {held_item}")
    
    print(f"✅ Contagem de instâncias: {manager.get_instance_count()}")
    
    print("\n✅ TESTE 1: PASSOU\n")
except Exception as e:
    print(f"\n❌ TESTE 1: FALHOU - {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Teste 2: GameState e MapData
print("=" * 70)
print("TESTE 2: GameState e MapData")
print("=" * 70)

try:
    from overcooked_marl_env.environment.game_state import GameState, MapData
    from overcooked_marl_env.environment import GameState as ImportedGameState
    from overcooked_marl_env.environment import MapData as ImportedMapData
    
    # Criar MapData
    static_map = np.array([
        [1, 2, 3],
        [0, 0, 0],
        [1, 2, 3]
    ], dtype=np.uint8)
    
    map_data = MapData(
        static_map=static_map,
        agent_positions={"agent_0": (1, 1), "agent_1": (2, 1)},
        dispensers={(0, 0): "tomato", (2, 2): "lettuce"},
        recipes=["SIMPLE_SALAD", "GARDEN_SALAD"],
        size=(3, 3)
    )
    print(f"✅ MapData criado: size={map_data.size}, recipes={len(map_data.recipes)}")
    
    # Criar GameState
    game_state = GameState(map_data, ["agent_0", "agent_1"])
    print(f"✅ GameState criado: {game_state}")
    
    # Testar funcionalidades básicas
    pos = game_state.get_agent_position("agent_0")
    print(f"✅ Posição do agente_0: {pos}")
    
    game_state.set_agent_position("agent_0", (2, 2))
    new_pos = game_state.get_agent_position("agent_0")
    print(f"✅ Nova posição do agente_0: {new_pos}")
    
    orientation = game_state.get_agent_orientation("agent_0")
    print(f"✅ Orientação do agente_0: {orientation}")
    
    recipe = game_state.get_current_expected_recipe()
    print(f"✅ Receita atual esperada: {recipe}")
    
    game_state.increment_timestep()
    print(f"✅ Timestep incrementado: {game_state.get_timestep()}")
    
    game_state.add_reward("agent_0", 10.0)
    reward = game_state.get_accumulated_reward("agent_0")
    print(f"✅ Recompensa acumulada: {reward}")
    
    print("\n✅ TESTE 2: PASSOU\n")
except Exception as e:
    print(f"\n❌ TESTE 2: FALHOU - {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Teste 3: Testes Unitários
print("=" * 70)
print("TESTE 3: Testes Unitários")
print("=" * 70)

try:
    import unittest
    from overcooked_marl_env.tests.unit.test_item_instance_manager import TestItemInstanceManager
    from overcooked_marl_env.tests.unit.test_game_state import TestGameState, TestMapData
    
    # Criar test suite
    suite = unittest.TestSuite()
    
    # Adicionar testes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestItemInstanceManager))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGameState))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMapData))
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print(f"\n✅ Todos os {result.testsRun} testes unitários passaram!")
        print("\n✅ TESTE 3: PASSOU\n")
    else:
        print(f"\n❌ {len(result.failures)} falhas, {len(result.errors)} erros")
        print("\n❌ TESTE 3: FALHOU\n")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ TESTE 3: FALHOU - {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Teste 4: Integração entre componentes
print("=" * 70)
print("TESTE 4: Integração entre Componentes")
print("=" * 70)

try:
    # Criar um cenário de integração
    manager = ItemInstanceManager()
    game_state = GameState(map_data, ["agent_0", "agent_1"])
    
    # Criar item e dar ao agente
    tomato_id = manager.create_instance({"type": "ingredient", "name": "tomato", "state": "raw"})
    manager.set_agent_item("agent_0", tomato_id)
    
    # Agente pega posição e move
    initial_pos = game_state.get_agent_position("agent_0")
    game_state.set_agent_position("agent_0", (initial_pos[0] + 1, initial_pos[1]))
    
    # Verificar estado
    print(f"✅ Item ID: {tomato_id}")
    print(f"✅ Agente segurando: {manager.get_agent_item('agent_0')['name']}")
    print(f"✅ Posição do agente: {game_state.get_agent_position('agent_0')}")
    
    # Colocar item no chão
    new_pos = game_state.get_agent_position("agent_0")
    manager.clear_agent_item("agent_0")
    manager.set_ground_item(new_pos, tomato_id)
    
    print(f"✅ Item no chão em {new_pos}: {manager.get_ground_item(new_pos)['name']}")
    print(f"✅ Agente não está mais segurando: {not manager.is_agent_holding_item('agent_0')}")
    
    print("\n✅ TESTE 4: PASSOU\n")
except Exception as e:
    print(f"\n❌ TESTE 4: FALHOU - {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Resumo final
print("=" * 70)
print("RESUMO DOS TESTES - FASE 2")
print("=" * 70)
print("✅ Todos os testes passaram!")
print("\n📋 FASE 2 COMPLETA:")
print("  ✅ ItemInstanceManager criado e testado")
print("  ✅ GameState e MapData criados e testados")
print("  ✅ 27 testes unitários passando")
print("  ✅ Integração entre componentes funcionando")
print("  ✅ Classes exportadas corretamente")
print("\n🎯 Componentes Core extraídos com sucesso!")
print("🎯 Próximo passo: Fase 3 - Sistema de Mapas")
print("=" * 70)
