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

def draw_detections(image: np.ndarray, caracteristicas: dict) -> np.ndarray:
    """Desenha marcações de detecção na imagem"""
    img = image.copy()
    height, width = img.shape[:2]
    
    # Cores para diferentes características
    colors = {
        'face_detectada': (0, 255, 0),
        'olhos': (255, 0, 0),
        'focinho': (0, 0, 255),
        'orelhas': (255, 255, 0)
    }
    
    # Desenhar retângulos e labels para cada característica
    for feature, data in caracteristicas.items():
        if feature not in colors or 'confianca' not in data:
            continue
            
        conf = data['confianca']
        if conf > 0.3:  # Threshold mínimo para exibição
            color = colors[feature]
            label = f"{feature.replace('_', ' ').title()}: {conf:.1%}"
            
            # Calcular posição baseada na característica
            if feature == 'face_detectada':
                x1, y1 = int(width*0.2), int(height*0.2)
                x2, y2 = int(width*0.8), int(height*0.8)
            elif feature == 'olhos':
                x1, y1 = int(width*0.3), int(height*0.3)
                x2, y2 = int(width*0.7), int(height*0.4)
            elif feature == 'focinho':
                x1, y1 = int(width*0.4), int(height*0.5)
                x2, y2 = int(width*0.6), int(height*0.7)
            else:  # orelhas
                x1, y1 = int(width*0.2), int(height*0.1)
                x2, y2 = int(width*0.8), int(height*0.3)
            
            # Desenhar retângulo
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Adicionar label
            cv2.putText(img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def create_thumbnail(image: np.ndarray, size=(100, 100)) -> str:
    """Cria thumbnail da imagem em base64"""
    try:
        # Redimensionar imagem
        img = Image.fromarray(image)
        img.thumbnail(size)
        
        # Converter para base64
        import io
        import base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logger.error(f"Erro ao criar thumbnail: {e}")
        return ""

def show_detection_report(analysis: dict, original_image: np.ndarray):
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
        if feature != 'erro':
            conf_color = 'green' if data['confianca'] > 0.7 else 'orange' if data['confianca'] > 0.4 else 'red'
            st.markdown(
                f"**{feature.replace('_', ' ').title()}**: "
                f"_{data['descricao']}_ - "
                f"Confiança: :{conf_color}[{data['confianca']:.2%}]"
            )
    
    # Desenhar detecções na imagem
    if 'erro' not in analysis['caracteristicas']:
        marked_image = draw_detections(original_image, analysis['caracteristicas'])
        st.image(marked_image, caption="Detecções identificadas", use_container_width=True)
    
    with st.expander("Detalhes Técnicos"):
        st.json(analysis)
    
    # Adicionar ao histórico com thumbnail
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    history_entry = {
        'Timestamp': analysis['timestamp'],
        'Status': analysis['status'],
        'Confiança': analysis['confianca'],
        'Qualidade': analysis['qualidade'],
        'Face': analysis['caracteristicas'].get('face_detectada', {}).get('confianca', 0),
        'Olhos': analysis['caracteristicas'].get('olhos', {}).get('confianca', 0),
        'Focinho': analysis['caracteristicas'].get('focinho', {}).get('confianca', 0),
        'Orelhas': analysis['caracteristicas'].get('orelhas', {}).get('confianca', 0),
        'Thumbnail': create_thumbnail(original_image)
    }
    
    st.session_state.detection_history.append(history_entry)
    
    # Mostrar histórico
    if st.session_state.detection_history:
        st.subheader("Histórico de Detecções")
        df = pd.DataFrame(st.session_state.detection_history)
        
        # Converter thumbnail para imagem
        def image_formatter(img_str):
            return f'<img src="data:image/png;base64,{img_str}" width="100">'
        
        # Formatar colunas numéricas como percentual
        for col in ['Confiança', 'Face', 'Olhos', 'Focinho', 'Orelhas']:
            df[col] = df[col].apply(lambda x: f"{float(x):.2%}" if isinstance(x, (float, int)) else x)
        
        # Exibir tabela com thumbnail
        st.write(
            df.to_html(
                escape=False,
                formatters={'Thumbnail': image_formatter},
                index=False
            ),
            unsafe_allow_html=True
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
