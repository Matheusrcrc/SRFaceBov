
import sqlite3
from datetime import datetime
import os

class BovineDatabase:
    def __init__(self, db_path='bovine_records.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            # Create tables
            c.executescript('''
                CREATE TABLE IF NOT EXISTS bovines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bovine_id TEXT UNIQUE,
                    name TEXT,
                    breed TEXT,
                    age INTEGER,
                    weight REAL,
                    notes TEXT,
                    registration_date DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS recognition_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bovine_id TEXT,
                    image_path TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (bovine_id) REFERENCES bovines (bovine_id)
                );
                
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bovine_id TEXT,
                    feature_vector BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (bovine_id) REFERENCES bovines (bovine_id)
                );
            ''')
    
    def add_bovine(self, bovine_id, name, breed, age, weight, notes=''):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO bovines (bovine_id, name, breed, age, weight, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (bovine_id, name, breed, age, weight, notes))
    
    def add_recognition_record(self, bovine_id, image_path, confidence):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO recognition_records (bovine_id, image_path, confidence)
                VALUES (?, ?, ?)
            ''', (bovine_id, image_path, confidence))
    
    def store_features(self, bovine_id, features):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO features (bovine_id, feature_vector)
                VALUES (?, ?)
            ''', (bovine_id, features.tobytes()))
    
    def get_bovine_info(self, bovine_id):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            return c.execute('''
                SELECT * FROM bovines WHERE bovine_id = ?
            ''', (bovine_id,)).fetchone()
    
    def get_recent_records(self, limit=10):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            return c.execute('''
                SELECT r.*, b.name, b.breed
                FROM recognition_records r
                LEFT JOIN bovines b ON r.bovine_id = b.bovine_id
                ORDER BY r.timestamp DESC
                LIMIT ?
            ''', (limit,)).fetchall()
    
    def get_all_features(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            return c.execute('''
                SELECT bovine_id, feature_vector
                FROM features
            ''').fetchall()
