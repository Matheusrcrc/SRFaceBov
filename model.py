
import tensorflow as tf
from tensorflow.keras import layers, models

def create_bovine_recognition_model():
    model = models.Sequential([
        # Base: MobileNetV2
        tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        ),
        
        # Additional layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # For binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == '__main__':
    # Create and save model
    model = create_bovine_recognition_model()
    model.save('bovine_recognition_model.h5')
    print("Model created and saved successfully")
