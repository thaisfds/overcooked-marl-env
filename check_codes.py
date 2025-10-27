from overcooked_marl_env.core.items import ItemManager

im = ItemManager()
print('CHAR_TO_NUM:')
for char, code in sorted(im.CHAR_TO_NUM.items(), key=lambda x: x[1]):
    print(f"  '{char}' = {code}")
