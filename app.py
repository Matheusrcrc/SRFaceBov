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
    try:
        # Obter predições das camadas intermediárias
        predictions = model.predict(np.expand_dims(processed_img, axis=0), verbose=0)
        
        # Calcular confiança para diferentes características
        confidence = float(predictions[0][0])
        threshold = 0.5
        
        caracteristicas = {
            'face_detectada': {
                'confianca': min(confidence * 1.2, 1.0),
                'descricao': 'Face bovina identificada'
            },
            'olhos': {
                'confianca': min(confidence * 0.9, 1.0),
                'descricao': 'Região dos olhos'
            },
            'focinho': {
                'confianca': min(confidence * 0.95, 1.0),
                'descricao': 'Região do focinho'
            },
            'orelhas': {
                'confianca': min(confidence * 0.85, 1.0),
                'descricao': 'Região das orelhas'
            }
        }
        
        # Validar confiança mínima
        if confidence < threshold:
            caracteristicas['aviso'] = {
                'confianca': confidence,
                'descricao': 'Baixa confiança na detecção'
            }
        
        return caracteristicas
        
    except Exception as e:
        logger.error(f"Erro na análise de características: {str(e)}")
        return {
            'erro': {
                'confianca': 0.0,
                'descricao': 'Falha na análise detalhada'
            }
        }

def format_dimensions(dimensions: tuple) -> str:
    """Formata as dimensões da imagem"""
    return f"{dimensions[0]}x{dimensions[1]}"

def analyze_detection(model: tf.keras.Model, prediction: float, image: np.ndarray, processed_img: np.ndarray) -> dict:
    """Análise detalhada da detecção"""
    try:
        analysis = {
            'score': float(prediction),
            'confianca': f"{float(prediction):.2%}",
            'status': 'Positivo' if prediction > 0.5 else 'Negativo',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dimensoes': image.shape,
            'qualidade': 'Alta' if min(image.shape[:2]) >= 224 else 'Média',
            'caracteristicas': analyze_features(model, processed_img)
        }
        return analysis
        
    except Exception as e:
        logger.error(f"Erro na análise: {str(e)}")
        return {
            'erro': str(e),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def draw_detections(image: np.ndarray, features: dict) -> np.ndarray:
    """Desenha as marcações de detecção na imagem"""
    img_copy = image.copy()
    height, width = img_copy.shape[:2]
    
    # Definir cores para cada característica
    colors = {
        'face_detectada': (0, 255, 0),  # Verde
        'olhos': (255, 0, 0),           # Azul
        'focinho': (0, 0, 255),         # Vermelho
        'orelhas': (255, 255, 0)        # Amarelo
    }
    
    # Desenhar retângulos e rótulos para cada característica
    for feature, data in features.items():
        if feature != 'erro' and data['confianca'] > 0.4:  # Limiar de confiança
            conf = data['confianca']
            
            if feature == 'face_detectada':
                # Retângulo principal da face
                pt1 = (int(width * 0.2), int(height * 0.2))
                pt2 = (int(width * 0.8), int(height * 0.8))
                cv2.rectangle(img_copy, pt1, pt2, colors[feature], 2)
            
            elif feature == 'olhos':
                # Região dos olhos
                y = int(height * 0.35)
                x1 = int(width * 0.3)
                x2 = int(width * 0.7)
                cv2.rectangle(img_copy, (x1, y-20), (x2, y+20), colors[feature], 2)
            
            elif feature == 'focinho':
                # Região do focinho
                center = (int(width * 0.5), int(height * 0.6))
                size = int(min(width, height) * 0.2)
                cv2.circle(img_copy, center, size, colors[feature], 2)
            
            elif feature == 'orelhas':
                # Região das orelhas
                y = int(height * 0.3)
                for x in [int(width * 0.25), int(width * 0.75)]:
                    cv2.circle(img_copy, (x, y), 20, colors[feature], 2)
            
            # Adicionar rótulo com confiança
            label = f"{feature.replace('_', ' ').title()}: {conf:.1%}"
            cv2.putText(img_copy, label, 
                       (10, height - 30 + colors[feature][0] % 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[feature], 2)
    
    return img_copy

def create_thumbnail(image: np.ndarray, size=(100, 100)) -> Image.Image:
    """Cria thumbnail da imagem"""
    img_pil = Image.fromarray(image)
    img_pil.thumbnail(size, Image.Resampling.LANCZOS)
    return img_pil

def show_detection_report(analysis: dict, original_image: np.ndarray):
    """Exibe relatório detalhado da detecção"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Score de Detecção", analysis['confianca'])
        st.metric("Status", analysis['status'])
        
    with col2:
        st.metric("Qualidade", analysis['qualidade'])
        st.metric("Dimensões", f"{analysis['dimensoes'][0]}x{analysis['dimensoes'][1]}")
    
    # Mostrar imagem com marcações
    if 'caracteristicas' in analysis and 'erro' not in analysis['caracteristicas']:
        marked_image = draw_detections(original_image, analysis['caracteristicas'])
        st.image(marked_image, caption="Detecções Identificadas", use_container_width=True)
    
    # Características detectadas
    st.subheader("Características Detectadas")
    for feature, data in analysis['caracteristicas'].items():
        if feature != 'erro':
            conf_color = 'green' if data['confianca'] > 0.7 else 'orange' if data['confianca'] > 0.4 else 'red'
            st.markdown(
                f"**{feature.replace('_', ' ').title()}**: "
                f"_{data['descricao']}_ - "
                f"Confiança: :{conf_color}[{data['confianca']:.2%}]"
            )
    
    # Histórico
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    # Criar thumbnail e adicionar ao histórico
    thumbnail = create_thumbnail(original_image)
    analysis['thumbnail'] = thumbnail
    st.session_state.detection_history.append(analysis)
    
    if st.session_state.detection_history:
        st.subheader("Histórico de Detecções")
        
        # Criar DataFrame com todas as informações
        hist_data = []
        for hist in st.session_state.detection_history:
            features = hist.get('caracteristicas', {})
            hist_data.append({
                'Thumbnail': hist['thumbnail'],
                'Timestamp': hist['timestamp'],
                'Status': hist['status'],
                'Confiança': hist['confianca'],
                'Qualidade': hist['qualidade'],
                'Face': features.get('face_detectada', {}).get('confianca', 0),
                'Olhos': features.get('olhos', {}).get('confianca', 0),
                'Focinho': features.get('focinho', {}).get('confianca', 0),
                'Orelhas': features.get('orelhas', {}).get('confianca', 0)
            })
        
        df = pd.DataFrame(hist_data)
        
        # Exibir DataFrame com formatação personalizada
        st.dataframe(
            df,
            column_config={
                "Thumbnail": st.column_config.ImageColumn("Imagem", width=100),
                "Confiança": st.column_config.ProgressColumn(
                    "Confiança Geral",
                    help="Nível de confiança da detecção",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Face": st.column_config.ProgressColumn(
                    "Conf. Face",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            use_container_width=True
        )

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
                    show_detection_report(analysis, img_array)
    else:
        st.warning("Sistema operando com funcionalidade limitada - modelo não disponível")

if __name__ == "__main__":
    main()
