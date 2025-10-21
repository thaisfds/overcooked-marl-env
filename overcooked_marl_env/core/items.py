from enum import IntEnum, auto

class ItemType(IntEnum):
    NONE = 0
    DISH = auto()
    
    # Itens Finais (Processados)
    TOMATO = auto()
    LETTUCE = auto()
    MEAT = auto()
    
    # Itens em Processamento
    TOMATO_RAW = auto()
    LETTUCE_RAW = auto()
    MEAT_RAW = auto()
    MEAT_CUT = auto()
    
    # Código de Receita Completa
    RECIPE_FULL = 15

class ItemManager:
    def __init__(self):
        self.CHAR_TO_NUM = {
            ' ': 0,
            '#': 1,
            'I': 2,
            'C': 3,
            'F': 4,
            'P': 5,
            'D': 6,
            'T': 7,
        }

        self.PROCESSING_MAP = {
            ItemType.TOMATO_RAW: ItemType.TOMATO,
            ItemType.LETTUCE_RAW: ItemType.LETTUCE,
            ItemType.MEAT_RAW: ItemType.MEAT_CUT,
            ItemType.MEAT_CUT: ItemType.MEAT,
        }

        self.STATION_PROCESSOR = {
            self.CHAR_TO_NUM['C']: [
                ItemType.TOMATO_RAW,
                ItemType.LETTUCE_RAW,
                ItemType.MEAT_RAW
            ],
            self.CHAR_TO_NUM['F']: [ItemType.MEAT_CUT],
        }

        self.PROCESSING_VECTOR_MAP = self._initialize_processing_vectors()

    def _initialize_processing_vectors(self):
        vectors = {
            ItemType.TOMATO_RAW: [1, 0, 0],
            ItemType.LETTUCE_RAW: [1, 0, 0],
            ItemType.MEAT_RAW: [1, 0, 0],
            ItemType.MEAT_CUT: [1, 1, 0],
            ItemType.TOMATO: [1, 1, 0],
            ItemType.LETTUCE: [1, 1, 0],
            ItemType.MEAT: [1, 1, 1],
            ItemType.DISH: [0, 0, 0],
            ItemType.NONE: [0, 0, 0],
        }
        
        # Adiciona códigos bit-wise (1-15)
        vectors.update({code: [0, 0, 0] for code in range(1, 16)})
        return vectors

    def get_item_code(self, item_type: ItemType) -> int:
        return item_type.value

    def get_cell_code(self, char: str) -> int:
        return self.CHAR_TO_NUM.get(char, 0)
