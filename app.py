
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import sqlite3
from datetime import datetime
import os

# Initialize database
def init_db():
    conn = sqlite3.connect('bovine_records.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS records
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         image_path TEXT,
         bovine_id TEXT,
         confidence REAL,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    ''')
    conn.commit()
    return conn

# Model loading function
@st.cache_resource
def load_model():
    # Placeholder for actual model loading
    # model = tf.keras.models.load_model('path_to_model')
    return None

# Image preprocessing
def preprocess_image(image):
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

# Initialize database
conn = init_db()

# Load model
model = load_model()

# Set page config
st.set_page_config(
    page_title="Sistema de Reconhecimento Facial Bovino",
    page_icon="🐮",
    layout="wide"
)

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
    
    # File uploader
    uploaded_file = st.file_uploader("Escolha uma imagem do bovino", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Imagem Original', use_column_width=True)
        
        if st.button('Processar Imagem'):
            # Preprocess image
            processed_img = preprocess_image(image)
            
            with col2:
                st.image(processed_img, caption='Imagem Processada', use_column_width=True)
            
            # Placeholder for model prediction
            st.info('Processando reconhecimento facial...')
            
            # Save record to database
            c = conn.cursor()
            image_path = f"images/{uploaded_file.name}"
            os.makedirs("images", exist_ok=True)
            image.save(image_path)
            
            # Simulated confidence score
            confidence = 0.95
            bovine_id = "BOV" + datetime.now().strftime("%Y%m%d%H%M%S")
            
            c.execute('''
                INSERT INTO records (image_path, bovine_id, confidence)
                VALUES (?, ?, ?)
            ''', (image_path, bovine_id, confidence))
            conn.commit()
            
            st.success(f'Bovino identificado com ID: {bovine_id}\nConfiança: {confidence:.2%}')

elif option == 'Visualizar Histórico':
    st.header("Histórico de Reconhecimentos")
    
    c = conn.cursor()
    records = c.execute('''
        SELECT bovine_id, confidence, timestamp, image_path
        FROM records
        ORDER BY timestamp DESC
        LIMIT 10
    ''').fetchall()
    
    if records:
        for record in records:
            bovine_id, confidence, timestamp, image_path = record
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if os.path.exists(image_path):
                    st.image(image_path, caption=f'ID: {bovine_id}', width=200)
            
            with col2:
                st.write(f"**ID do Bovino:** {bovine_id}")
                st.write(f"**Confiança:** {confidence:.2%}")
                st.write(f"**Data:** {timestamp}")
                st.markdown("---")
    else:
        st.info("Nenhum registro encontrado")

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
            st.success("Bovino cadastrado com sucesso!")

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
