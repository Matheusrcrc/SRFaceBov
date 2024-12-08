
import pytest
from utils import preprocess_image, calculate_similarity
import numpy as np
from PIL import Image

def test_preprocess_image():
    # Create a test image
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3)).astype('uint8'))
    
    # Test preprocessing
    processed = preprocess_image(img)
    
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0

def test_calculate_similarity():
    # Create test features
    features1 = np.random.random((1, 128))
    features2 = np.random.random((1, 128))
    
    similarity = calculate_similarity(features1, features2)
    
    assert isinstance(similarity, float)
    assert -1.0 <= similarity <= 1.0

if __name__ == '__main__':
    pytest.main([__file__])
