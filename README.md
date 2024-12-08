# Sistema de Reconhecimento Facial Bovino

## Visão Geral
Sistema de reconhecimento facial bovino baseado em IA para identificação e rastreamento de bovinos em ambientes de produção. Utiliza YOLOv8 e TensorFlow para detecção e reconhecimento preciso.

## Características
- Detecção facial bovina em tempo real
- Reconhecimento e identificação individual
- Interface web intuitiva com Streamlit
- Sistema de monitoramento de recursos
- Banco de dados integrado para rastreamento
- Suporte a imagens e vídeos
- Processamento em GPU (quando disponível)

## Requisitos Técnicos
- Python 3.12+
- CUDA (opcional, para GPU)
- Sistema operacional: Linux, macOS ou Windows
- Mínimo de 4GB RAM (8GB recomendado)
- Espaço em disco: 2GB+

## Instalação

1. Criar ambiente virtual:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Install required packages
pip install -r requirements.txt
```

## Uso
1. Configuração do Banco de Dados
```python
from bovine_system import BovineDatabase
db = BovineDatabase()
db.initialize_database()
```

2. Processamento de Imagem
```python
from bovine_system import BovineImageProcessor
processor = BovineImageProcessor()
features = processor.process_image('bovine_image.jpg')
```

3. Reconhecimento
```python
from bovine_system import BovineRecognition
recognizer = BovineRecognition()
result = recognizer.identify_bovine('test_image.jpg')
```

## Estrutura do Projeto
```
bovine_recognition/
├── data/
│   └── bovine.db
├── images/
│   └── processed/
├── logs/
├── models/
└── output/
```

## Métricas de Desempenho
- Precisão na Detecção Facial: 95%
- Limite de Confiança para Reconhecimento: 0.85
- Tempo Médio de Processamento: <2s por imagem

## Licença
MIT License

## Contribuição
1. Faça um fork do repositório
2. Crie sua branch de feature
3. Envie um pull request com uma descrição abrangente

## Suporte
Para suporte, entre em contato com support@bovinerecognition.com
