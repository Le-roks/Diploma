import streamlit as st
import pandas as pd
import base64
from io import BytesIO

# --- ІМПОРТ МОДУЛІВ ---
from modules.preprocessor import prepare_image
from modules.inference import load_model_file, predict_image
from modules.report_generator import generate_csv_report, get_report_filename

# --- НАЛАШТУВАННЯ СТОРІНКИ ---
st.set_page_config(
    page_title="Ідентифікація пошкоджень",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ПІДКЛЮЧЕННЯ СТИЛІВ ---
def local_css(file_name):
    try:
        with open(file_name, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Файл {file_name} не знайдено!")

local_css("style/style.css") 

# --- ФУНКЦІЯ ДЛЯ КОНВЕРТАЦІЇ ФОТО В ТАБЛИЦЮ ---
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# --- ЗАВАНТАЖЕННЯ МОДЕЛІ ---
MODEL_PATH = "models/mobile_net_v2.h5" 
model = load_model_file(MODEL_PATH)

if model is None:
    st.error("❌ Помилка: файл моделі не знайдено! Перевірте папку models/")
    st.stop()

# --- ІНІЦІАЛІЗАЦІЯ СЕСІЇ ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'single_result' not in st.session_state:
    st.session_state.single_result = None

# --- SIDEBAR (Меню зліва) ---
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
            st.session_state.results_df = None
            st.session_state.single_result = None
            
            progress_bar = st.progress(0, text="Аналіз зображень...")
            temp_results = []
            
            for i, file in enumerate(uploaded_files):
                img_tensor, original_img = prepare_image(file)
                label, conf = predict_image(model, img_tensor)
                
                # Підготовка мініатюри
                thumb = original_img.copy()
                thumb.thumbnail((120, 120)) 
                img_base64 = image_to_base64(thumb)

                temp_results.append({
                    "Фото": img_base64,
                    "Файл": file.name if hasattr(file, 'name') else "Camera",
                    "Зображення_Original": original_img,
                    "Клас": label,
                    # множимо на 100 тут для UI 
                    "Впевненість": float(conf) * 100 
                })
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            progress_bar.empty()
            
            if len(temp_results) == 1:
                st.session_state.single_result = temp_results[0]
            else:
                df_clean = []
                for r in temp_results:
                    df_clean.append({
                        "Фото": r["Фото"],
                        "Файл": r["Файл"],
                        "Клас": r["Клас"],
                        "Впевненість": r["Впевненість"]
                    })
                st.session_state.results_df = pd.DataFrame(df_clean)


# --- MAIN AREA (Основна частина екрану) ---
st.title("🍎 Ідентифікація пошкоджень плодовоовочевої продукції")
st.divider()

# ЛОГІКА ВІДОБРАЖЕННЯ

# 1. СЦЕНАРІЙ 1: ОДИН ФАЙЛ
if st.session_state.single_result:
    res = st.session_state.single_result
    
    c1, c2 = st.columns([1, 2], gap="large", vertical_alignment="center")
    
    with c1:
        st.image(res["Зображення_Original"], caption="Вхідне зображення", use_container_width=True)
        
    with c2:
        st.subheader("Результат класифікації")
        if res["Клас"] == "Пошкоджений":
            st.error(f"⚠️ Виявлено ознаки пошкодження")
        else:
            st.success(f"✅ Дефектів не виявлено")
        
        st.metric("Рівень впевненості", f"{res['Впевненість']:.2f}%")
        st.progress(res['Впевненість'] / 100)

# 2. СЦЕНАРІЙ 2: ПАКЕТНИЙ РЕЖИМ (ТАБЛИЦЯ)
elif st.session_state.results_df is not None:
    df = st.session_state.results_df
    
    total = len(df)
    rotten = len(df[df["Клас"] == "Пошкоджений"])
    healthy = total - rotten
    
    st.markdown("### 📊 Статистика партії")
    
    # Вставляємо HTML-код карток з вашими змінними (total, rotten, healthy)
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-card">
            <span class="stat-label">Всього файлів</span>
            <span class="stat-value value-neutral">{total}</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Пошкоджених</span>
            <span class="stat-value value-error">{rotten}</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Здорових</span>
            <span class="stat-value value-success">{healthy}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    st.subheader("📋 Детальний звіт")

    # Стилізація
    def style_dataframe(df):
        base_styler = df.style.set_properties(**{
            'text-align': 'center',
            'vertical-align': 'middle',
            'font-weight': '500'
        })
        
        def color_class(val):
            style = 'font-weight: bold; ' 
            if val == 'Пошкоджений':
                return style + 'color: #d32f2f; background-color: #ffebee;'
            elif val == 'Здоровий':
                return style + 'color: #2e7d32; background-color: #e8f5e9;'
            return style

        return base_styler.map(color_class, subset=['Клас'])

    styled_df = style_dataframe(df)

    st.dataframe(
        styled_df,
        use_container_width=True,
        row_height=100,
        column_order=["Фото", "Файл", "Клас", "Впевненість"],
        column_config={
            "Фото": st.column_config.ImageColumn("Фото", width="small"),
            "Файл": st.column_config.TextColumn("Файл", width="large"),
            "Клас": st.column_config.TextColumn("Клас", width="small"),
            "Впевненість": st.column_config.NumberColumn("Впевненість (%)", format="%.2f %%", width="small")
        }
    )
    
    # --- ЕКСПОРТ ---
    # Створюємо копію для експорту
    df_export = df.drop(columns=['Фото']).copy()
    
    # Форматуємо число 99.24 у рядок "99.24%"
    # Це гарантує, що в CSV буде правильний вигляд і Excel не домножить це ще раз
    df_export['Впевненість'] = df_export['Впевненість'].apply(lambda x: f"{x:.2f}%")
    
    csv_text = generate_csv_report(df_export.to_dict('records'))
    csv_bytes = csv_text.encode('utf-8-sig')
    report_filename = get_report_filename()
    
    st.download_button(
        label="📥 Завантажити звіт (CSV)",
        data=csv_bytes,
        file_name=report_filename,
        mime='text/csv',
        type="primary"
    )

# 3. НОВИЙ СТАН
elif uploaded_files:
    st.info("✅ Зображення обрано! \n\n👈 Натисніть кнопку **«Виконати класифікацію»** у меню зліва, щоб отримати результат.")

# 4. ПОЧАТКОВИЙ СТАН
else:
    st.info("👈 Завантажте зображення через меню зліва для початку роботи.")

# --- ФУТЕР ---
st.markdown("""
    <div class="footer-container">
        <p>© 2025 Терещенко В. С. | Розроблено в рамках магістерської роботи</p>
    </div>
""", unsafe_allow_html=True)