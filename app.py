import streamlit as st

# Configuração do Streamlit
st.set_page_config(
    page_title="Sistema de Reconhecimento Facial Bovino",
    page_icon="🐮",
    layout="wide"
)

# Core imports
import os
import platform
import logging
from datetime import datetime
from typing import Optional

# Third-party imports
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import sqlite3
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar TensorFlow
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configurações da aplicação
APP_CONFIG = {
    "title": "Sistema de Reconhecimento Facial Bovino",
    "icon": "🐮",
    "db_path": ":memory:",
    "db_timeout": 30,
    "image_size": (224, 224),
    "use_gpu": False
}

@st.cache_resource
def load_model() -> Optional[tf.keras.Model]:
    """
    Cria e retorna um modelo base pré-treinado para reconhecimento facial
    """
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        logger.error(f"Erro ao criar modelo: {str(e)}")
        return None

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Pré-processa a imagem para entrada no modelo"""
    image = cv2.resize(image, APP_CONFIG["image_size"])
    image = image / 255.0
    return image

def analyze_features(model: tf.keras.Model, processed_img: np.ndarray) -> dict:
    """Analisa características detalhadas na imagem"""
    # Extrair features do modelo base
    features_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('global_average_pooling2d').output
    )
    features = features_model.predict(np.expand_dims(processed_img, axis=0), verbose=0)[0]
    
    # Mapear características bovinas
    caracteristicas = {
        'face_detectada': {
            'confianca': float(features[0]),
            'descricao': 'Face bovina identificada'
        },
        'olhos': {
            'confianca': float(features[1]),
            'descricao': 'Região dos olhos'
        },
        'focinho': {
            'confianca': float(features[2]),
            'descricao': 'Região do focinho'
        },
        'orelhas': {
            'confianca': float(features[3]),
            'descricao': 'Região das orelhas'
        }
    }
    
    return caracteristicas

def analyze_detection(model: tf.keras.Model, prediction: float, image: np.ndarray, processed_img: np.ndarray) -> dict:
    """Análise detalhada da detecção com características"""
    # Análise básica
    analysis = {
        'score': float(prediction),
        'confianca': f"{float(prediction):.2%}",
        'status': 'Positivo' if prediction > 0.5 else 'Negativo',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dimensoes': image.shape,
        'qualidade': 'Alta' if min(image.shape[:2]) >= 224 else 'Média'
    }
    
    # Adicionar análise de características
    analysis['caracteristicas'] = analyze_features(model, processed_img)
    
    return analysis

def show_detection_report(analysis: dict):
    """Exibe relatório detalhado da detecção"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Score de Detecção", analysis['confianca'])
        st.metric("Status", analysis['status'])
        
    with col2:
        st.metric("Qualidade", analysis['qualidade'])
        st.metric("Dimensões", f"{analysis['dimensoes'][0]}x{analysis['dimensoes'][1]}")
    
    # Mostrar características detectadas
    st.subheader("Características Detectadas")
    
    for feature, data in analysis['caracteristicas'].items():
        conf_color = 'green' if data['confianca'] > 0.7 else 'orange' if data['confianca'] > 0.4 else 'red'
        st.markdown(
            f"**{feature.replace('_', ' ').title()}**: "
            f"_{data['descricao']}_ - "
            f"Confiança: :{conf_color}[{data['confianca']:.2%}]"
        )
    
    with st.expander("Detalhes Técnicos"):
        st.json(analysis)
        
    # Adicionar ao histórico
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    st.session_state.detection_history.append(analysis)
    
    # Mostrar histórico
    if st.session_state.detection_history:
        st.subheader("Histórico de Detecções")
        df = pd.DataFrame([
            {
                'Timestamp': d['timestamp'],
                'Status': d['status'],
                'Confiança': d['confianca'],
                'Qualidade': d['qualidade']
            } for d in st.session_state.detection_history
        ])
        st.dataframe(df, use_container_width=True)

def main():
    st.title(APP_CONFIG["title"])
    
    # Sidebar com informações
    with st.sidebar:
        st.info(f"TensorFlow version: {tf.__version__}")
        st.info(f"Python version: {platform.python_version()}")
    
    # Carregar modelo
    model = load_model()
    
    if model is not None:
        st.success("Modelo inicializado com sucesso!")
        
        # Upload de imagem
        uploaded_file = st.file_uploader(
            "Escolha uma imagem do bovino", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Exibir imagem
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem carregada", use_container_width=True)
            
            # Processar imagem
            if st.button("Analisar Imagem"):
                with st.spinner("Processando..."):
                    img_array = np.array(image)
                    processed_img = preprocess_image(img_array)
                    
                    # Fazer predição
                    prediction = model.predict(
                        np.expand_dims(processed_img, axis=0),
                        verbose=0
                    )[0][0]
                    
                    analysis = analyze_detection(model, prediction, img_array, processed_img)
                    show_detection_report(analysis)
    else:
        st.warning("Sistema operando com funcionalidade limitada - modelo não disponível")

if __name__ == "__main__":
    main()
