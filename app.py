# Core imports primeiro
import os
import platform
import logging
from datetime import datetime
from typing import Optional

# Third-party imports em segundo
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import sqlite3

# Configurar logging antes de tudo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações da aplicação
APP_CONFIG = {
    "title": "Sistema de Reconhecimento Facial Bovino",
    "icon": "🐮",
    "db_path": os.path.join(os.path.dirname(__file__), "bovine_records.db"),
    "db_timeout": 30
}

# Configuração Streamlit
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["icon"],
    layout="wide"
)

# Verificar ambiente
tf_version = tf.__version__
python_version = platform.python_version()

# Informações no sidebar
with st.sidebar:
    st.info(f"TensorFlow version: {tf_version}")
    st.info(f"Python version: {python_version}")

def init_db() -> Optional[sqlite3.Connection]:
    """
    Inicializa a conexão com o banco de dados SQLite.
    
    Returns:
        Optional[sqlite3.Connection]: Conexão com o banco ou None se houver erro
    """
    try:
        conn = sqlite3.connect(
            APP_CONFIG["db_path"],
            timeout=APP_CONFIG["db_timeout"]
        )
        logger.info(f"Banco de dados conectado: {APP_CONFIG['db_path']}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Erro ao conectar ao banco: {e}")
        st.error(f"Erro de conexão com banco de dados: {e}")
        return None

# Initialize database with improved error handling
def init_db() -> Optional[sqlite3.Connection]:
    """
    Inicializa a conexão com o banco de dados SQLite.
    
    Returns:
        Optional[sqlite3.Connection]: Objeto de conexão ou None se houver erro
    """
    try:
        conn = sqlite3.connect(
            APP_CONFIG["db_path"], 
            timeout=APP_CONFIG["db_timeout"]
        )
        c = conn.cursor()
        
        # Criar tabela se não existir
        c.execute('''
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                bovine_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(image_path, bovine_id)
            )
        ''')
        conn.commit()
        logging.info("Banco de dados inicializado com sucesso")
        return conn
        
    except sqlite3.Error as e:
        logging.error(f"Erro SQLite: {e}")
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return None
        
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
        st.error(f"Erro inesperado: {e}")
        return None

# Model loading function with improved error handling
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        # Placeholder for actual model loading
        # Adicione aqui o caminho correto do seu modelo
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.h5')
        if not os.path.exists(model_path):
            st.error("Modelo não encontrado. Verifique o caminho do arquivo.")
            return None
            
        model = tf.keras.models.load_model(model_path)
        return model
    except tf.errors.NotFoundError:
        st.error("Erro: Arquivo do modelo não encontrado")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Image preprocessing with error handling
def preprocess_image(image):
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Resize image
        img_array = cv2.resize(img_array, (224, 224))
        
        # Normalize pixel values
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Erro no pré-processamento da imagem: {e}")
        return None

def main():
    try:
        # Initialize database
        conn = init_db()
        if conn is None:
            st.error("Não foi possível inicializar o banco de dados")
            return

        # Load model
        model = load_model()

        # Title and description
        st.title("Sistema de Reconhecimento Facial Bovino")
        st.markdown("Sistema para identificação e reconhecimento facial de bovinos usando IA")

        # Sidebar
        st.sidebar.header("Opções")
        option = st.sidebar.selectbox(
            'Escolha uma operação:',
            ('Upload de Imagem', 'Visualizar Histórico', 'Cadastro de Bovinos', 'Sobre o Sistema')
        )

        if option == 'Upload de Imagem':
            st.header("Upload de Imagem")
            
            uploaded_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption='Imagem carregada', use_column_width=True)
                    
                    if st.button('Processar Imagem'):
                        processed_img = preprocess_image(image)
                        if processed_img is not None:
                            # Placeholder for actual processing
                            st.success("Imagem processada com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao processar a imagem: {e}")

        elif option == 'Visualizar Histórico':
            st.header("Histórico de Identificações")
            
            try:
                c = conn.cursor()
                c.execute("SELECT * FROM records ORDER BY timestamp DESC")
                records = c.fetchall()
                
                if records:
                    for record in records:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            image_path = record[1]
                            if os.path.exists(image_path):
                                st.image(image_path, caption=f'ID: {record[2]}', width=200)
                        
                        with col2:
                            st.write(f"**ID do Bovino:** {record[2]}")
                            st.write(f"**Confiança:** {record[3]:.2%}")
                            st.write(f"**Data:** {record[4]}")
                            st.markdown("---")
                else:
                    st.info("Nenhum registro encontrado")
            except Exception as e:
                st.error(f"Erro ao carregar histórico: {e}")

        elif option == 'Cadastro de Bovinos':
            st.header("Cadastro de Novos Bovinos")
            
            with st.form("cadastro_bovino"):
                nome = st.text_input("Nome/Identificação do Bovino")
                raca = st.selectbox("Raça", ["Nelore", "Angus", "Brahman", "Hereford", "Outro"])
                idade = st.number_input("Idade (meses)", min_value=0, max_value=360)
                peso = st.number_input("Peso (kg)", min_value=0.0)
                observacoes = st.text_area("Observações")
                
                submitted = st.form_submit_button("Cadastrar")
                
                if submitted:
                    try:
                        # Implementar lógica de cadastro aqui
                        st.success("Bovino cadastrado com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao cadastrar bovino: {e}")

        else:  # Sobre o Sistema
            st.header("Sobre o Sistema")
            st.markdown('''
            ### Funcionalidades principais:
            - Reconhecimento facial de bovinos usando IA
            - Registro e acompanhamento de identificações
            - Cadastro e gerenciamento de rebanho
            - Interface intuitiva para usuários
            
            ### Como utilizar:
            1. Selecione "Upload de Imagem" no menu lateral
            2. Faça upload da foto do bovino
            3. Clique em "Processar Imagem"
            4. Aguarde o resultado do processamento
            
            ### Tecnologias utilizadas:
            - TensorFlow para processamento de imagem
            - OpenCV para pré-processamento
            - SQLite para armazenamento de dados
            - Streamlit para interface do usuário
            ''')

        # Footer
        st.markdown("---")
        st.markdown("Desenvolvido com ❤️ para identificação bovina")

    except Exception as e:
        st.error(f"Erro geral na aplicação: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
