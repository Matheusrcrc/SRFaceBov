
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

def detect_bovine_face(image):
    '''
    Detect bovine face in the image using Haar Cascade or similar technique
    Returns the cropped face region
    '''
    # Convert PIL Image to cv2 format
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # TODO: Implement actual face detection
    # For now, return the center region of the image
    h, w = gray.shape
    center_x, center_y = w // 2, h // 2
    size = min(w, h) // 2
    
    face_region = img_array[
        max(0, center_y - size):min(h, center_y + size),
        max(0, center_x - size):min(w, center_x + size)
    ]
    
    return Image.fromarray(face_region)

def preprocess_for_model(image, target_size=(224, 224)):
    '''
    Preprocess image for model input
    '''
    # Resize
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def extract_features(image, model):
    '''
    Extract features from preprocessed image using the model
    '''
    preprocessed = preprocess_for_model(image)
    features = model.predict(preprocessed)
    return features

def compare_features(features1, features2, threshold=0.85):
    '''
    Compare two feature vectors and return similarity score
    '''
    similarity = np.dot(features1.flatten(), features2.flatten()) /                 (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity > threshold, similarity
