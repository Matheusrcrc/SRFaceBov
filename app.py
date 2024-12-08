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
def load_model() -> Optional[YOLO]:
    """Carrega modelo YOLO para detecção"""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        logger.info("Modelo YOLO carregado com sucesso")
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo YOLO: {e}")
        return None

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Pré-processa a imagem para entrada no modelo"""
    image = cv2.resize(image, APP_CONFIG["image_size"])
    image = image / 255.0
    return image

def process_image(image: np.ndarray, model: YOLO) -> dict:
    """Processa imagem com YOLO e retorna análise"""
    try:
        # Fazer predição
        results = model(image)[0]
        
        # Extrair detecções
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            detection = {
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confianca': float(conf),
                'classe': results.names[int(cls)]
            }
            detections.append(detection)
            
        # Preparar análise
        analysis = {
            'score': max([d['confianca'] for d in detections], default=0.0),
            'confianca': f"{max([d['confianca'] for d in detections], default=0.0):.2%}",
            'status': 'Positivo' if detections else 'Negativo',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dimensoes': image.shape,
            'qualidade': 'Alta' if min(image.shape[:2]) >= 640 else 'Média',
            'caracteristicas': {
                f"deteccao_{i}": {
                    'confianca': d['confianca'],
                    'descricao': f"Detectado {d['classe']}"
                }
                for i, d in enumerate(detections)
            }
        }
        
        # Desenhar detecções
        annotated_frame = results.plot()
        analysis['imagem_processada'] = annotated_frame
        
        return analysis
        
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        return {
            'erro': str(e),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    """Desenha marcações precisas de detecção na imagem"""
    try:
        # Garantir que a imagem está no formato correto
        img = image.copy()
        if len(img.shape) == 2:  # Se for grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # Se tiver canal alpha
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
        height, width = img.shape[:2]
        
        # Configurações visuais
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(width, height) / 1000
        thickness = max(1, int(min(width, height) / 500))
        
        # Definição das ROIs
        rois = {
            'face_detectada': {
                'region': [
                    int(width*0.25), int(height*0.15),
                    int(width*0.75), int(height*0.85)
                ],
                'color': (0, 255, 0),
                'label_pos': 'top'
            },
            'olhos': {
                'region': [
                    int(width*0.35), int(height*0.25),
                    int(width*0.65), int(height*0.35)
                ],
                'color': (255, 0, 0),
                'label_pos': 'top'
            },
            'focinho': {
                'region': [
                    int(width*0.45), int(height*0.45),
                    int(width*0.55), int(height*0.60)
                ],
                'color': (0, 0, 255),
                'label_pos': 'bottom'
            },
            'orelhas': {
                'region': [
                    int(width*0.30), int(height*0.15),
                    int(width*0.70), int(height*0.25)
                ],
                'color': (255, 255, 0),
                'label_pos': 'top'
            }
        }
        
        for feature, data in caracteristicas.items():
            if feature not in rois or 'confianca' not in data:
                continue
                
            conf = data['confianca']
            if conf > 0.3:
                roi = rois[feature]
                color = roi['color']
                x1, y1, x2, y2 = roi['region']
                
                # Desenhar retângulo
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                
                # Preparar texto
                label = f"{feature.replace('_', ' ').title()}: {conf:.1%}"
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                
                # Posicionar texto
                if roi['label_pos'] == 'top':
                    text_y = max(text_size[1] + 10, y1 - 5)
                else:
                    text_y = min(height - 10, y2 + text_size[1] + 5)
                    
                text_x = x1 + (x2 - x1 - text_size[0]) // 2
                text_x = max(5, min(text_x, width - text_size[0] - 5))
                
                # Desenhar fundo do texto
                cv2.rectangle(img,
                            (text_x - 2, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + 2),
                            (0, 0, 0),
                            -1)
                
                # Desenhar texto
                cv2.putText(img, label,
                          (text_x, text_y),
                          font, font_scale, color, thickness)
        
        return img
        
    except Exception as e:
        logger.error(f"Erro ao desenhar detecções: {str(e)}")
        return image  # Retorna imagem original em caso de erro

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
    """Exibe relatório com detecções YOLO"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Score de Detecção", analysis['confianca'])
        st.metric("Status", analysis['status'])
        
    with col2:
        st.metric("Qualidade", analysis['qualidade'])
        st.metric("Dimensões", f"{analysis['dimensoes'][0]}x{analysis['dimensoes'][1]}")
    
    # Mostrar detecções
    if 'imagem_processada' in analysis:
        st.image(analysis['imagem_processada'], caption="Detecções YOLO", use_container_width=True)
    
    # Características detectadas
    st.subheader("Detecções")
    for feature, data in analysis['caracteristicas'].items():
        conf_color = 'green' if data['confianca'] > 0.7 else 'orange' if data['confianca'] > 0.4 else 'red'
        st.markdown(
            f"**{feature}**: {data['descricao']} - "
            f"Confiança: :{conf_color}[{data['confianca']:.2%}]"
        )
    
    with st.expander("Detalhes Técnicos"):
        st.json(analysis)
        
    # Atualizar histórico
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    history_entry = {
        'Timestamp': analysis['timestamp'],
        'Status': analysis['status'],
        'Confiança': analysis['confianca'],
        'Qualidade': analysis['qualidade'],
        'Detecções': len(analysis['caracteristicas']),
        'Thumbnail': create_thumbnail(original_image)
    }
    
    st.session_state.detection_history.append(history_entry)
    
    # Mostrar histórico
    if st.session_state.detection_history:
        st.subheader("Histórico de Detecções")
        df = pd.DataFrame(st.session_state.detection_history)
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
                    show_detection_report(analysis, img_array)
    else:
        st.warning("Sistema operando com funcionalidade limitada - modelo não disponível")

if __name__ == "__main__":
    main()
