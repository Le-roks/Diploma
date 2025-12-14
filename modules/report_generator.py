import pandas as pd
from datetime import datetime, timedelta

def generate_csv_report(data_list: list) -> str:
    """
    Перетворює підготовлені дані у формат CSV тексту.
    """
    # Створюємо DataFrame з готового списку
    df = pd.DataFrame(data_list)
    
    # Просто повертаємо CSV рядок (utf-8-sig для коректного відкриття в Excel)
    return df.to_csv(index=False, encoding='utf-8-sig')

def get_report_filename() -> str:
    """
    Генерує назву файлу з поточним київським часом.
    """
    # UTC + 2 години (Київський час)
    kyiv_time = datetime.utcnow() + timedelta(hours=2)
    
    # Формат: report_2025-12-12_14-30-00.csv
    return f"report_{kyiv_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"