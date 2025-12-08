import streamlit as st
import pandas as pd
import time

# Імпорт модулів
from modules.preprocessor import prepare_image
from modules.inference import load_model_file, predict_image

# --- НАЛАШТУВАННЯ СТОРІНКИ ---
st.set_page_config(
    page_title="Ідентифікація пошкоджень",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ПІДКЛЮЧЕННЯ СТОРІННИХ СТИЛІВ ---
def local_css(file_name):
    try:
        with open(file_name, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Файл {file_name} не знайдено!")

local_css("style/style.css") # Викликаємо наш файл стилів

# --- ЗАВАНТАЖЕННЯ МОДЕЛІ ---
MODEL_PATH = "models/mobile_net_v2.h5" 
model = load_model_file(MODEL_PATH)

if model is None:
    st.error("❌ Помилка: Файл моделі не знайдено! Перевірте папку models/")
    st.stop()

# --- ІНІЦІАЛІЗАЦІЯ СЕСІЇ (Щоб дані не зникали) ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'single_result' not in st.session_state:
    st.session_state.single_result = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("Налаштування вводу")
    source = st.radio("Джерело даних:", ["Завантаження файлів", "Використати камеру"])
    
    uploaded_files = []
    if source == "Завантаження файлів":
        uploaded_files = st.file_uploader(
            "Оберіть зображення", 
            type=['png', 'jpg', 'jpeg'], 
            accept_multiple_files=True
        )
    elif source == "Використати камеру":
        cam_file = st.camera_input("Зробити фото")
        if cam_file:
            uploaded_files = [cam_file]

    st.markdown("---")
    
    # Кнопка запуску
    if st.button("Виконати класифікацію", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Спочатку завантажте файли!")
        else:
            # Очищення
            st.session_state.results_df = None
            st.session_state.single_result = None
            
            progress_bar = st.progress(0, text="Аналіз зображень...")
            temp_results = []
            
            for i, file in enumerate(uploaded_files):
                img_tensor, original_img = prepare_image(file)
                label, conf = predict_image(model, img_tensor)
                
                temp_results.append({
                    "Файл": file.name if hasattr(file, 'name') else "Camera",
                    "Зображення": original_img,
                    "Клас": label,
                    "Впевненість": float(conf)
                })
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            progress_bar.empty()
            
            # Збереження в сесію
            if len(temp_results) == 1:
                st.session_state.single_result = temp_results[0]
            else:
                df_clean = []
                for r in temp_results:
                    df_clean.append({
                        "Файл": r["Файл"],
                        "Клас": r["Клас"],
                        "Впевненість": r["Впевненість"]
                    })
                st.session_state.results_df = pd.DataFrame(df_clean)


# --- MAIN AREA ---
st.title("🍎 Система ідентифікації пошкоджень")

# СЦЕНАРІЙ 1: ОДИН ФАЙЛ
if st.session_state.single_result:
    res = st.session_state.single_result
    st.divider()
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.image(res["Зображення"], caption="Вхідний об'єкт", width=300)
        
    with c2:
        st.subheader("Результат діагностики")
        if res["Клас"] == "Пошкоджений":
            st.error(f"⚠️ Виявлено: {res['Клас']}")
        else:
            st.success(f"✅ Виявлено: {res['Клас']}")
        
        st.metric("Рівень впевненості", f"{res['Впевненість']*100:.2f}%")
        st.progress(res['Впевненість'])

# СЦЕНАРІЙ 2: ПАКЕТНИЙ РЕЖИМ
elif st.session_state.results_df is not None:
    df = st.session_state.results_df
    
    # Статистика
    total = len(df)
    rotten = len(df[df["Клас"] == "Пошкоджений"])
    healthy = total - rotten
    
    st.markdown("### 📊 Статистика партії")
    m1, m2, m3 = st.columns(3)
    m1.metric("Всього", total)
    m2.metric("Пошкоджених", rotten, delta_color="inverse")
    m3.metric("Здорових", healthy, delta_color="normal")
    
    st.divider()
    st.subheader("📋 Детальний звіт")

    # --- ДИЗАЙН ТАБЛИЦІ (З чіткими межами) ---
    def badge_style(val):
        if val == 'Пошкоджений':
            # Додано border (рамку)
            return 'background-color: #ffebee; color: #c62828; font-weight: bold; border: 1px solid #ffcdd2; border-radius: 4px;'
        elif val == 'Здоровий':
            # Додано border (рамку)
            return 'background-color: #e8f5e9; color: #2e7d32; font-weight: bold; border: 1px solid #c8e6c9; border-radius: 4px;'
        return ''

    # Застосовуємо стиль
    styled_df = df.style.map(badge_style, subset=['Клас']).format("{:.2%}", subset=['Впевненість'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # --- CSV (Кодування для Excel) ---
    # utf-8-sig додає BOM, щоб Excel зрозумів кирилицю
    csv = df.to_csv(index=False).encode('utf-8-sig')
    
    st.download_button(
        label="📥 Завантажити звіт (CSV)",
        data=csv,
        file_name='identification_report.csv',
        mime='text/csv',
        type="primary"
    )

elif not uploaded_files:
    st.info("👈 Завантажте зображення через меню зліва для початку роботи.")