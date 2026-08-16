import os
import sys

try:
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from huggingface_hub import login
except ImportError:
    print("Устанавливаю необходимые библиотеки...")
    os.system(f"{sys.executable} -m pip install datasets huggingface_hub pandas")
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from huggingface_hub import login

# Добавляем корневую директорию в PYTHONPATH чтобы найти synthfin_aml_pkg
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synthfin_aml_pkg.generator import FraudGraphGenerator

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ОШИБКА: Токен Hugging Face не найден.")
        print("Пожалуйста, установите переменную окружения HF_TOKEN перед запуском.")
        print("В PowerShell: $env:HF_TOKEN=\"ваш_токен_с_huggingface_hub\"")
        print("Токен можно получить здесь (нужны права Write): https://huggingface.co/settings/tokens")
        sys.exit(1)
        
    print("Логинимся в Hugging Face...")
    login(token=token, add_to_git_credential=True)
    
    print("Генерируем датасет V9.1 (100k узлов, 10 дней)...")
    # Используем seed=42 для воспроизводимости
    gen = FraudGraphGenerator(seed=42)
    # Генерируем объемный датасет для HF
    gen.generate_transactions(agents=100000, days=10)
    
    nodes_df, edges_df = gen.to_dataframes()
    
    print(f"Сгенерировано: Узлов={len(nodes_df)}, Ребер={len(edges_df)}")
    
    repo_id = "ovvaliyev/synthfin-aml"
    print(f"Загружаем узлы (config='nodes') в репозиторий {repo_id}...")
    ds_nodes = Dataset.from_pandas(nodes_df)
    ds_nodes.push_to_hub(repo_id, config_name="nodes", private=False)
    
    print(f"Загружаем ребра (config='edges') в репозиторий {repo_id}...")
    ds_edges = Dataset.from_pandas(edges_df)
    ds_edges.push_to_hub(repo_id, config_name="edges", private=False)
    print("✅ Успешно загружено! Датасет теперь публично доступен.")

if __name__ == "__main__":
    main()
