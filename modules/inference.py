import tensorflow as tf
import streamlit as st
import numpy as np

# Кешування, щоб не вантажити модель 50 разів для 50 файлів
@st.cache_resource
def load_model_file(path_to_model):
    try:
        model = tf.keras.models.load_model(path_to_model)
        return model
    except Exception as e:
        st.error(f"Критична помилка: Не вдалося завантажити файл моделі! {e}")
        return None

def predict_image(model, img_tensor):
    """
    Повертає текстову мітку класу та відсоток впевненості.
    """
    if model is None:
        return "Помилка", 0.0

    # Отримуємо передбачення (наприклад: [[0.05, 0.95]])
    prediction = model.predict(img_tensor, verbose=0)
    
    # Індекс 0 = Healthy (Здоровий)
    # Індекс 1 = Rotten (Пошкоджений)
    
    classes = ['Здоровий', 'Пошкоджений'] 
    
    # Якщо Sigmoid (один вихід):
    if prediction.shape[-1] == 1:
        score = prediction[0][0]
        if score > 0.5:
            label = classes[1] # Rotten
            confidence = score
        else:
            label = classes[0] # Healthy
            confidence = 1 - score
            
    # Якщо Softmax (два виходи):
    else:
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        label = classes[class_index]
    
    return label, confidence