import numpy as np
from PIL import Image

def prepare_image(uploaded_file):
    """
    Повертає:
    1. img_tensor - для моделі (224x224)
    2. original_image - для відображення на екрані (без спотворень)
    """
    # 1. Відкриваємо оригінал
    original_image = Image.open(uploaded_file).convert('RGB')
    
    # 2. Робимо копію для моделі і змінюємо її розмір
    # (Використовуємо копію, щоб не зіпсувати оригінал для екрану)
    model_input = original_image.copy().resize((224, 224))
    
    # 3. Перетворюємо в масив та нормалізуємо
    img_array = np.array(model_input) / 255.0
    
    # 4. Додаємо batch dimension
    img_tensor = np.expand_dims(img_array, axis=0)
    
    return img_tensor, original_image