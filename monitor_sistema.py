
import os
import time
import psutil
import logging
from datetime import datetime

# Configuração do logging
logging.basicConfig(
    filename='sistema_monitoramento.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def monitorar_sistema():
    while True:
        try:
            # Uso de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Uso de memória
            mem = psutil.virtual_memory()
            
            # Uso de disco
            disk = psutil.disk_usage('/')
            
            # Processos do Streamlit
            streamlit_processes = [p for p in psutil.process_iter(['pid', 'name'])
                                 if 'streamlit' in p.info['name'].lower()]
            
            # Registrar informações
            logging.info(f"CPU: {cpu_percent}%")
            logging.info(f"Memória: {mem.percent}%")
            logging.info(f"Disco: {disk.percent}%")
            logging.info(f"Processos Streamlit: {len(streamlit_processes)}")
            
            # Verificar limites
            if cpu_percent > 80:
                logging.warning("Uso de CPU alto!")
            if mem.percent > 80:
                logging.warning("Uso de memória alto!")
            if disk.percent > 80:
                logging.warning("Uso de disco alto!")
            
            time.sleep(60)  # Verificar a cada minuto
            
        except Exception as e:
            logging.error(f"Erro no monitoramento: {str(e)}")
            time.sleep(60)  # Aguardar um minuto antes de tentar novamente

if __name__ == "__main__":
    logging.info("Sistema de monitoramento iniciado")
    monitorar_sistema()
