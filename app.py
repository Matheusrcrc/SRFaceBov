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
from typing import Optional, Any

# Importações com tratamento de erro
try:
    import cv2
except ImportError:
    print("Erro ao importar cv2, tentando opencv-python...")
    os.system('pip install opencv-python')
    try:
        import cv2
    except ImportError:
        print("Erro ao importar cv2, tentando opencv-python-headless...")
        os.system('pip install opencv-python-headless')
        import cv2

# Third-party imports
import numpy as np
import tensorflow as tf
import sqlite3
import pandas as pd
from ultralytics import YOLO  # Importação do YOLO

# Adicionar após as importações
import torch

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'  # Corrigido de 'mensagem' para 'message'
)
logger = logging.getLogger(__name__)

# Configurar TensorFlow
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configurações da aplicação
APP_CONFIG = {
    "title": "Sistema de Reconhecimento Facial Bovino",
    "icon": "🐮",
    "db_path": "bovinos.db",  # Alterado para arquivo físico
    "db_timeout": 30,
    "image_size": (224, 224),
    "use_gpu": False,
    "confidence": 0.25,
    "model_path": "yolov8n.pt",
    "available_models": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
}

def get_db():
    """Retorna conexão com banco de dados"""
    try:
        conn = sqlite3.connect(APP_CONFIG["db_path"], timeout=APP_CONFIG["db_timeout"])
        c = conn.cursor()
        
        # Criar tabela se não existir
        c.execute('''
            CREATE TABLE IF NOT EXISTS bovinos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                codigo TEXT UNIQUE,
                nome TEXT,
                caracteristicas TEXT,
                confianca REAL,
                primeira_deteccao DATETIME,
                ultima_deteccao DATETIME,
                total_deteccoes INTEGER DEFAULT 1
            )
        ''')
        conn.commit()
        return conn
    except sqlite3.Error as e:
        logger.error(f"Erro banco de dados: {e}")
        return None

# Adicionar após as configurações
def init_db():
    """Inicializa banco de dados SQLite"""
    try:
        conn = sqlite3.connect(APP_CONFIG["db_path"])
        c = conn.cursor()
        
        # Tabela principal
        c.execute('''
            CREATE TABLE IF NOT EXISTS bovinos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                codigo TEXT UNIQUE,
                nome TEXT,
                caracteristicas TEXT,
                confianca REAL,
                primeira_deteccao DATETIME,
                ultima_deteccao DATETIME,
                total_deteccoes INTEGER DEFAULT 1
            )
        ''')
        
        # Tabela de histórico
        c.execute('''
            CREATE TABLE IF NOT EXISTS historico_deteccoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bovino_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confianca REAL,
                detalhes TEXT,
                FOREIGN KEY (bovino_id) REFERENCES bovinos (id)
            )
        ''')
        
        conn.commit()
        return conn
    except Exception as e:
        logger.error(f"Erro ao inicializar banco: {e}")
        return None

def find_similar_bovino(caracteristicas: dict, threshold: float = 0.85) -> Optional[dict]:
    """Busca bovino similar no banco"""
    try:
        conn = get_db()
        if not conn:
            return None
            
        c = conn.cursor()
        c.execute("SELECT * FROM bovinos")
        bovinos = c.fetchall()
        
        for bovino in bovinos:
            stored_caract = eval(bovino[3])
            similarity = calculate_similarity(caracteristicas, stored_caract)
            if similarity >= threshold:
                return {
                    'id': bovino[0],
                    'codigo': bovino[1],
                    'nome': bovino[2],
                    'similarity': similarity
                }
        return None
    except Exception as e:
        logger.error(f"Erro ao buscar bovino: {e}")
        return None
    finally:
        if conn:
            conn.close()

def calculate_similarity(caract1: dict, caract2: dict) -> float:
    """Calcula similaridade entre características"""
    try:
        # Comparar features principais
        features1 = [v['confianca'] for k, v in caract1.items()]
        features2 = [v['confianca'] for k, v in caract2.items()]
        
        # Calcular similaridade usando correlação
        return float(np.corrcoef(features1, features2)[0, 1])
    except:
        return 0.0

def register_new_bovino(caracteristicas: dict, nome: str) -> Optional[dict]:
    """Registra novo bovino apenas com nome"""
    try:
        conn = get_db()
        if not conn:
            return None
            
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Gerar código automaticamente baseado no timestamp
        codigo = f"BOV_{now.replace(' ', '').replace('-', '').replace(':', '')}"
        
        c.execute("""
            INSERT INTO bovinos 
            (codigo, nome, caracteristicas, confianca, primeira_deteccao, ultima_deteccao)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            codigo, 
            nome, 
            str(caracteristicas),
            max(v['confianca'] for v in caracteristicas.values()),
            now,
            now
        ))
        
        bovino_id = c.lastrowid  # Pegar ID do bovino inserido
        conn.commit()
        
        # Salvar primeira detecção no histórico
        save_detection_history(bovino_id, {
            'score': max(v['confianca'] for v in caracteristicas.values()),
            'caracteristicas': caracteristicas
        })
        
        return {'codigo': codigo, 'nome': nome, 'id': bovino_id}
    except Exception as e:
        logger.error(f"Erro ao registrar bovino: {e}")
        st.error(f"Erro ao registrar: {e}")
        return None
    finally:
        if conn:
            conn.close()

def update_bovino_detection(bovino_id: int) -> bool:
    """Atualiza contagem de detecções"""
    try:
        conn = get_db()
        if not conn:
            return False
            
        c = conn.cursor()
        c.execute("""
            UPDATE bovinos 
            SET total_deteccoes = total_deteccoes + 1,
                ultima_deteccao = ?
            WHERE id = ?
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), bovino_id))
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Erro ao atualizar detecção: {e}")
        return False
    finally:
        if conn:
            conn.close()

@st.cache_resource
def load_model(model_path: str = "yolov8n.pt") -> Optional[YOLO]:  # Adicionado parâmetro model_path
    """Carrega modelo YOLO para detecção"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)  # Usa o model_path fornecido
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
    """Processa imagem com YOLO"""
    try:
        # Configurar parâmetros YOLO e fazer predição
        results = model.predict(
            source=image,
            conf=0.25,
            show=False
        )
        
        # Processar resultados
        detections = []
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                box_data = box.data[0]
                detection = {
                    'bbox': box_data[:4].tolist(),
                    'confianca': float(box_data[4]),
                    'classe': results[0].names[int(box_data[5])]
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
        
        # Adicionar imagem processada
        if len(results) > 0:
            analysis['imagem_processada'] = results[0].plot()
        
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
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Corrigido formato da string
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

def create_thumbnail(image: np.ndarray, size=(100, 100)) -> np.ndarray:
    """Cria thumbnail da imagem"""
    try:
        # Usar cv2 ao invés de PIL
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.error(f"Erro ao criar thumbnail: {e}")
        return image

def save_detection_history(bovino_id: int, analysis: dict) -> bool:
    """Salva histórico de uma detecção"""
    try:
        conn = get_db()
        if not conn:
            return False
            
        c = conn.cursor()
        c.execute("""
            INSERT INTO historico_deteccoes 
            (bovino_id, timestamp, confianca, detalhes)
            VALUES (?, ?, ?, ?)
        """, (
            bovino_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            analysis['score'],
            str(analysis)
        ))
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar histórico: {e}")
        return False
    finally:
        if conn:
            conn.close()

def show_detection_report(analysis: dict, image_np: np.ndarray):
    """Exibe relatório com identificação do bovino"""
    if 'caracteristicas' not in analysis:
        st.error("Nenhuma característica detectada na imagem")
        return

    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Mostrar imagem processada
        if 'imagem_processada' in analysis:
            st.image(
                analysis['imagem_processada'],
                caption="Detecções",
                use_container_width=True
            )
    
    with col2:
        # Métricas principais
        st.metric("Confiança", analysis['confianca'])
        st.metric("Status", analysis['status'])
        st.metric("Qualidade", analysis['qualidade'])
        
        # Buscar bovino similar
        similar = find_similar_bovino(analysis['caracteristicas'])
        
        if similar:
            # Bovino conhecido
            st.success("🐮 Bovino Reconhecido!")
            st.write(f"**Nome:** {similar['nome']}")
            st.write(f"**Similaridade:** {similar['similarity']:.2%}")
            
            # Atualizar registros e salvar histórico
            if update_bovino_detection(similar['id']):
                save_detection_history(similar['id'], analysis)
                st.info("✅ Registro atualizado com sucesso!")
                
            # Mostrar histórico
            show_bovino_history(similar['id'])
        else:
            st.warning("⚠️ Novo bovino detectado!")
            
            # Formulário para registro simplificado
            nome = st.text_input("Nome do bovino:")
            if st.button("Registrar Bovino"):
                if nome:
                    novo_bovino = register_new_bovino(
                        analysis['caracteristicas'],
                        nome
                    )
                    if novo_bovino:
                        st.success(f"✅ Bovino {novo_bovino['nome']} registrado!")
                        # Mostrar histórico do novo bovino
                        show_bovino_history(novo_bovino['id'])
                    else:
                        st.error("❌ Erro ao registrar bovino")
                else:
                    st.error("Por favor, informe o nome do bovino")

def show_bovino_history(bovino_id: int):
    """Mostra histórico de detecções do bovino"""
    try:
        conn = get_db()
        if not conn:
            return
            
        # Buscar dados do histórico
        df = pd.read_sql_query('''
            SELECT 
                strftime('%d/%m/%Y %H:%M', h.timestamp) as Data,
                h.confianca as Confiança,
                COUNT(*) OVER (ORDER BY h.timestamp) as "Detecção #"
            FROM historico_deteccoes h
            WHERE h.bovino_id = ?
            ORDER BY h.timestamp DESC
            LIMIT 10
        ''', conn, params=(bovino_id,))
        
        if not df.empty:
            st.write("### Histórico de Detecções")
            
            # Formatar colunas
            df['Confiança'] = df['Confiança'].apply(lambda x: f"{x:.1%}")
            
            # Exibir tabela formatada
            st.dataframe(
                df,
                column_config={
                    "Data": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm"),
                    "Confiança": st.column_config.ProgressColumn(min_value=0, max_value=100),
                    "Detecção #": st.column_config.NumberColumn(format="%d")
                },
                hide_index=True,
                use_container_width=True
            )
        
    except Exception as e:
        logger.error(f"Erro ao mostrar histórico: {e}")
    finally:
        if conn:
            conn.close()

def main():
    st.title(APP_CONFIG["title"])
    
    # Inicializar banco de dados primeiro
    if not os.path.exists(APP_CONFIG["db_path"]):
        init_db()
        
    # Sidebar para configurações
    with st.sidebar:
        # ...existing code...
        
        st.header("Configurações")
        confidence = st.slider(
            "Confiança", 
            min_value=0.0, 
            max_value=1.0, 
            value=APP_CONFIG["confidence"]
        )
        
        model_path = st.selectbox(
            "Selecione o modelo",
            APP_CONFIG["available_models"]
        )
        
        source_type = st.radio(
            "Tipo de entrada",
            ["Imagem", "Vídeo"]
        )

    # Carregar modelo com base na seleção
    model = load_model(model_path)
    if model is None:
        st.error("Erro ao carregar modelo YOLO")
        return

    if source_type == "Imagem":
        uploaded_file = st.file_uploader(
            "Escolha uma imagem do bovino", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file:
            # Ler imagem com cv2
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            # Converter BGR para RGB para exibição no Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Imagem carregada", use_container_width=True)
            
            if st.button("Analisar Imagem"):
                with st.spinner("Processando..."):
                    processed_img = preprocess_image(image)
                    analysis = process_image(image, model)
                    
                    if 'erro' not in analysis:
                        show_detection_report(analysis, image)
                    else:
                        st.error(f"Erro ao processar imagem: {analysis['erro']}")
            
    else:  # Vídeo
        uploaded_video = st.file_uploader(
            "Escolha um vídeo", 
            type=["mp4", "avi", "mov"]
        )
        
        if uploaded_video:
            # Salvar vídeo temporariamente
            temp_path = f"temp_{uploaded_video.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            # Processar vídeo
            try:
                with st.spinner("Processando vídeo..."):
                    results = model.track(
                        source=temp_path,
                        conf=confidence,
                        persist=True
                    )
                    
                    # Mostrar vídeo processado
                    processed_video = results[0].save(temp_path)
                    st.video(processed_video)
                    
            except Exception as e:
                st.error(f"Erro ao processar vídeo: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # Inicializar banco de dados
    if not os.path.exists(APP_CONFIG["db_path"]):
        init_db()
    
    # ...rest of main() function...

if __name__ == "__main__":
    main()
