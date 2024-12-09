import sqlite3
import logging
from datetime import datetime
import json

class DatabaseManager:
    def __init__(self, db_path='bovine_records.db'):
        self.db_path = db_path
        self.setup_logging()
        self.initialize_db()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('database.log'),
                logging.StreamHandler()
            ]
        )
    
    def initialize_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Tabela principal de registros
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS bovine_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bovine_id TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        features BLOB,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de histórico de reconhecimento
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recognition_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bovine_id TEXT,
                        confidence REAL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logging.info('Database initialized successfully')
        except Exception as e:
            logging.error(f'Error initializing database: {str(e)}')
    
    def add_bovine_record(self, bovine_id, image_path, features=None, metadata=None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO bovine_records (bovine_id, image_path, features, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (bovine_id, image_path, features, json.dumps(metadata) if metadata else None))
                conn.commit()
                logging.info(f'Added new record for bovine ID: {bovine_id}')
                return cursor.lastrowid
        except Exception as e:
            logging.error(f'Error adding bovine record: {str(e)}')
            return None
    
    def get_bovine_record(self, bovine_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM bovine_records WHERE bovine_id = ?', (bovine_id,))
                return cursor.fetchone()
        except Exception as e:
            logging.error(f'Error retrieving bovine record: {str(e)}')
            return None
    
    def add_recognition_event(self, bovine_id, confidence):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO recognition_history (bovine_id, confidence)
                    VALUES (?, ?)
                ''', (bovine_id, confidence))
                conn.commit()
                logging.info(f'Added recognition event for bovine ID: {bovine_id}')
        except Exception as e:
            logging.error(f'Error adding recognition event: {str(e)}')
    
    def get_recognition_history(self, bovine_id=None, limit=100):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if bovine_id:
                    cursor.execute('''
                        SELECT * FROM recognition_history 
                        WHERE bovine_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (bovine_id, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM recognition_history 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (limit,))
                return cursor.fetchall()
        except Exception as e:
            logging.error(f'Error retrieving recognition history: {str(e)}')
            return []
