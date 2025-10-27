"""
Script de teste para validar a Fase 1 da refatoração.

Este script testa se a estrutura básica foi criada corretamente.
"""

import sys
import os

# Teste 1: Imports de constantes
print("=" * 60)
print("TESTE 1: Imports de Constantes")
print("=" * 60)

try:
    from overcooked_marl_env.core.constants import (
        MAX_PROCESS_STEPS,
        TOTAL_CHANNELS,
        NUM_BASE_CHANNELS,
        DEFAULT_NUM_AGENTS,
        MAX_EPISODE_STEPS,
        ACTION_UP,
        ACTION_DOWN,
        ACTION_LEFT,
        ACTION_RIGHT,
        ACTION_INTERACT,
        get_agent_color,
        get_station_color,
        get_direction_name,
    )
    
    print(f"✅ MAX_PROCESS_STEPS: {MAX_PROCESS_STEPS}")
    print(f"✅ TOTAL_CHANNELS: {TOTAL_CHANNELS}")
    print(f"✅ NUM_BASE_CHANNELS: {NUM_BASE_CHANNELS}")
    print(f"✅ DEFAULT_NUM_AGENTS: {DEFAULT_NUM_AGENTS}")
    print(f"✅ MAX_EPISODE_STEPS: {MAX_EPISODE_STEPS}")
    print(f"✅ Ações: UP={ACTION_UP}, DOWN={ACTION_DOWN}, LEFT={ACTION_LEFT}, RIGHT={ACTION_RIGHT}, INTERACT={ACTION_INTERACT}")
    print(f"✅ Funções utilitárias importadas com sucesso")
    print("\n✅ TESTE 1: PASSOU\n")
except ImportError as e:
    print(f"❌ TESTE 1: FALHOU - {e}\n")
    sys.exit(1)

# Teste 2: Verificar estrutura de diretórios
print("=" * 60)
print("TESTE 2: Estrutura de Diretórios")
print("=" * 60)

# Get the actual package path
import overcooked_marl_env
env_path = os.path.dirname(overcooked_marl_env.__file__)

required_dirs = [
    "environment",  # Renamed from 'env' to avoid conflict with env.py
    "map",
    "map/maps",
    "mechanics",
    "mechanics/interactions",
    "core",
    "rendering",
    "utils"
]

all_exist = True
for dir_name in required_dirs:
    full_path = os.path.join(env_path, dir_name)
    exists = os.path.isdir(full_path)
    symbol = "✅" if exists else "❌"
    print(f"{symbol} {dir_name}/")
    if not exists:
        all_exist = False

if all_exist:
    print("\n✅ TESTE 2: PASSOU\n")
else:
    print("\n❌ TESTE 2: FALHOU - Alguns diretórios não foram criados\n")
    sys.exit(1)

# Teste 3: Verificar arquivos de mapa
print("=" * 60)
print("TESTE 3: Arquivos de Mapa")
print("=" * 60)

map_files = ["default_map.txt", "tutorial.txt"]
maps_path = os.path.join(env_path, "map", "maps")

all_maps_exist = True
for map_file in map_files:
    full_path = os.path.join(maps_path, map_file)
    exists = os.path.isfile(full_path)
    symbol = "✅" if exists else "❌"
    print(f"{symbol} {map_file}")
    if not exists:
        all_maps_exist = False

if all_maps_exist:
    print("\n✅ TESTE 3: PASSOU\n")
else:
    print("\n❌ TESTE 3: FALHOU - Alguns mapas não foram copiados\n")
    sys.exit(1)

# Teste 4: Verificar __init__.py files
print("=" * 60)
print("TESTE 4: Arquivos __init__.py")
print("=" * 60)

init_files = [
    "environment/__init__.py",  # Renamed from 'env' to avoid conflict
    "map/__init__.py",
    "mechanics/__init__.py",
    "mechanics/interactions/__init__.py",
    "core/__init__.py",
    "rendering/__init__.py",
    "utils/__init__.py"
]

all_inits_exist = True
for init_file in init_files:
    full_path = os.path.join(env_path, init_file)
    exists = os.path.isfile(full_path)
    symbol = "✅" if exists else "❌"
    print(f"{symbol} {init_file}")
    if not exists:
        all_inits_exist = False

if all_inits_exist:
    print("\n✅ TESTE 4: PASSOU\n")
else:
    print("\n❌ TESTE 4: FALHOU - Alguns __init__.py não foram criados\n")
    sys.exit(1)

# Teste 5: Testar funções utilitárias
print("=" * 60)
print("TESTE 5: Funções Utilitárias")
print("=" * 60)

try:
    # Testar get_agent_color
    color_0 = get_agent_color(0)
    color_1 = get_agent_color(1)
    print(f"✅ get_agent_color(0): {color_0}")
    print(f"✅ get_agent_color(1): {color_1}")
    
    # Testar get_direction_name
    north = get_direction_name((0, -1))
    south = get_direction_name((0, 1))
    print(f"✅ get_direction_name((0, -1)): {north}")
    print(f"✅ get_direction_name((0, 1)): {south}")
    
    # Testar get_station_color
    from overcooked_marl_env.core.constants import CELL_CHOPPING_BOARD
    color = get_station_color(CELL_CHOPPING_BOARD)
    print(f"✅ get_station_color('C'): {color}")
    
    print("\n✅ TESTE 5: PASSOU\n")
except Exception as e:
    print(f"❌ TESTE 5: FALHOU - {e}\n")
    sys.exit(1)

# Resumo final
print("=" * 60)
print("RESUMO DOS TESTES")
print("=" * 60)
print("✅ Todos os testes passaram!")
print("\n📋 FASE 1 COMPLETA:")
print("  ✅ Estrutura de diretórios criada")
print("  ✅ Arquivo de constantes funcionando")
print("  ✅ Mapas copiados para novo local")
print("  ✅ Arquivos __init__.py configurados")
print("  ✅ Funções utilitárias operacionais")
print("\n🎯 Próximo passo: Fase 2 - Extração de Componentes Core")
print("=" * 60)
