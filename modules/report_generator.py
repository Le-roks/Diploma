"""
Модуль генерації звітності (Report Generator)
Відповідає за формування таблиць результатів та експорт у CSV формат.
"""
import pandas as pd
from datetime import datetime
from typing import List, Dict
from datetime import datetime, timedelta


def generate_results_table(results_data: List[Dict]) -> pd.DataFrame:
    """
    Формує зведену таблицю результатів для відображення та експорту.
    
    Args:
        results_data: Список словників з результатами обробки
                     [{'Файл': str, 'Клас': str, 'Впевненість': float, 
                       'Ймовірність_Здоровий': float, 'Ймовірність_Пошкоджений': float}, ...]
        
    Returns:
        pd.DataFrame: Таблиця з результатами
    """
    df = pd.DataFrame(results_data)
    
    # Переупорядковуємо колонки для кращого відображення
    column_order = ['Файл', 'Клас', 'Впевненість', 
                    'Ймовірність_Здоровий', 'Ймовірність_Пошкоджений']
    
    # Додаємо тільки ті колонки, які є в даних
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]
    
    # Форматуємо відсотки для кращої читабельності
    if 'Впевненість' in df.columns:
        df['Впевненість'] = df['Впевненість'].apply(lambda x: f"{x*100:.2f}%")
    if 'Ймовірність_Здоровий' in df.columns:
        df['Ймовірність_Здоровий'] = df['Ймовірність_Здоровий'].apply(lambda x: f"{x*100:.2f}%")
    if 'Ймовірність_Пошкоджений' in df.columns:
        df['Ймовірність_Пошкоджений'] = df['Ймовірність_Пошкоджений'].apply(lambda x: f"{x*100:.2f}%")
    
    return df


def generate_csv_report(results_data: List[Dict], include_timestamp: bool = True) -> str:
    """
    Генерує CSV звіт з результатами ідентифікації.
    
    Args:
        results_data: Список словників з результатами обробки
        include_timestamp: Чи додавати часову мітку до назви файлу
        
    Returns:
        str: CSV рядок у форматі UTF-8
    """
    # Створюємо DataFrame
    df = pd.DataFrame(results_data)
    
    # Видаляємо колонку зображення, якщо вона є (не можна експортувати в CSV)
    if 'Зображення' in df.columns:
        df = df.drop(columns=['Зображення'])
    
    # Форматуємо числові значення як відсотки
    numeric_columns = ['Впевненість', 'Ймовірність_Здоровий', 'Ймовірність_Пошкоджений']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x)
    
    # Конвертуємо в CSV
    csv_string = df.to_csv(index=False, encoding='utf-8-sig')  # utf-8-sig для Excel
    
    return csv_string


def get_report_filename() -> str:
    """
    Генерує назву файлу звіту з часовою міткою.
    
    Returns:
        str: Назва файлу у форматі identification_report_YYYYMMDD_HHMMSS.csv
    """

    kyiv_time = datetime.utcnow() + timedelta(hours=2)
    return f"identification_report_{kyiv_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"


def calculate_statistics(results_data: List[Dict]) -> Dict:
    """
    Обчислює статистику по результатах обробки.
    
    Args:
        results_data: Список словників з результатами
        
    Returns:
        dict: {
            'total': int,
            'healthy_count': int,
            'damaged_count': int,
            'healthy_percentage': float,
            'damaged_percentage': float
        }
    """
    total = len(results_data)
    damaged_count = sum(1 for r in results_data if r.get('Клас') == 'Пошкоджений')
    healthy_count = total - damaged_count
    
    return {
        'total': total,
        'healthy_count': healthy_count,
        'damaged_count': damaged_count,
        'healthy_percentage': (healthy_count / total * 100) if total > 0 else 0,
        'damaged_percentage': (damaged_count / total * 100) if total > 0 else 0
    }

