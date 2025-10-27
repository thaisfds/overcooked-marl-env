from overcooked_marl_env.map.map_parser import MapParser

parser = MapParser()
result = parser.parse_file('overcooked_marl_env/map/maps/tutorial.txt')

print(f'Grid shape: {result["grid"].shape}')
print(f'Size: {result["size"]}')
print(f'Agents: {result["agent_positions"]}')
print(f'Dispensers: {result["dispensers"]}')
print(f'\nGrid (first 10 rows):')
for y in range(min(10, result["grid"].shape[0])):
    print(f'Row {y}: {result["grid"][y,:]}')
