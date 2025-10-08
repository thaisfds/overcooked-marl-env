from setuptools import setup, find_packages

setup(
    # Nome do pacote (será usado para o pip install)
    name="overcooked-marl-env",
    
    # Versão inicial do seu ambiente
    version="0.0.1",
    
    # Informações básicas
    description="Um ambiente Gymnasium/PettingZoo para MARL baseado em Overcooked.",
    author="[Seu Nome ou Usuário]",
    
    # Encontra automaticamente a sua pasta de código ('overcooked_marl_env')
    packages=find_packages(),
    
    # Dependências necessárias para rodar o ambiente
    install_requires=[
        "gymnasium>=0.29.1", # A versão mais recente do Gymnasium
        "numpy",             # Para lidar com arrays e observações
        "pettingzoo>=1.24.0",# Essencial para a classe base ParallelEnv
    ],
    
    # Versão mínima do Python
    python_requires=">=3.10", # Recomendado usar Python 3.10+
)