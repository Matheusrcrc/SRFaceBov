import cv2
import numpy as np
from PIL import Image
import logging
import os
from datetime import datetime

def setup_logging(name='utils'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Garante formato correto
        handlers=[
            logging.FileHandler(f'{name}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

logger = setup_logging()

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f'Created directory: {path}')

def save_image(image, directory='images', prefix='bovine'):
    ensure_directory(directory)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{prefix}_{timestamp}.jpg'
    filepath = os.path.join(directory, filename)
    
    if isinstance(image, np.ndarray):
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elif isinstance(image, Image.Image):
        image.save(filepath)
    else:
        raise ValueError('Unsupported image type')
    
    logger.info(f'Saved image to: {filepath}')
    return filepath

def load_image(filepath):
    try:
        image = Image.open(filepath)
        return image
    except Exception as e:
        logger.error(f'Error loading image {filepath}: {str(e)}')
        return None

def preprocess_image(image, target_size=(224, 224)):
    if isinstance(image, str):
        image = load_image(image)
    if image is None:
        return None
        
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    return img_array

def calculate_similarity(features1, features2):
    return np.dot(features1.flatten(), features2.flatten()) / (
        np.linalg.norm(features1.flatten()) * np.linalg.norm(features2.flatten())
    )
