
import psutil
import time
import logging
from datetime import datetime
import os
import json

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_monitor.log'),
        logging.StreamHandler()
    ]
)

class SystemMonitor:
    def __init__(self):
        self.stats = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'processes': []
        }
        
    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self):
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used
        }
    
    def get_disk_usage(self):
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': disk.percent
        }
    
    def get_process_info(self):
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                if 'streamlit' in pinfo['name'].lower() or 'python' in pinfo['name'].lower():
                    processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return processes
    
    def monitor(self, interval=60):
        while True:
            try:
                current_stats = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu': self.get_cpu_usage(),
                    'memory': self.get_memory_usage(),
                    'disk': self.get_disk_usage(),
                    'processes': self.get_process_info()
                }
                
                # Log stats
                logging.info(json.dumps(current_stats))
                
                # Save to stats history
                self.stats['cpu'].append(current_stats['cpu'])
                self.stats['memory'].append(current_stats['memory']['percent'])
                self.stats['disk'].append(current_stats['disk']['percent'])
                self.stats['processes'].append(len(current_stats['processes']))
                
                # Keep only last 24 hours of data (assuming 1-minute intervals)
                max_entries = 24 * 60
                for key in self.stats:
                    if len(self.stats[key]) > max_entries:
                        self.stats[key] = self.stats[key][-max_entries:]
                
                # Alert if resources are running low
                if current_stats['cpu'] > 80:
                    logging.warning('High CPU usage: %s%%', current_stats['cpu'])
                if current_stats['memory']['percent'] > 80:
                    logging.warning('High memory usage: %s%%', current_stats['memory']['percent'])
                if current_stats['disk']['percent'] > 80:
                    logging.warning('High disk usage: %s%%', current_stats['disk']['percent'])
                
                time.sleep(interval)
                
            except Exception as e:
                logging.error('Error in monitoring: %s', str(e))
                time.sleep(interval)

if __name__ == '__main__':
    monitor = SystemMonitor()
    monitor.monitor()
