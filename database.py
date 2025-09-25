import sqlite3
from datetime import datetime

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('wardrobe_wizard.db', check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create try_on_history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS try_on_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT NOT NULL,
            style_id TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Create favorites table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            style_id TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        self.conn.commit()
    
    def add_user(self, username, email):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (username, email) VALUES (?, ?)',
                (username, email)
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None
    
    def add_try_on(self, user_id, image_path, style_id):
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO try_on_history (user_id, image_path, style_id) VALUES (?, ?, ?)',
            (user_id, image_path, style_id)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def add_favorite(self, user_id, style_id):
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO favorites (user_id, style_id) VALUES (?, ?)',
            (user_id, style_id)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def get_user_history(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT * FROM try_on_history WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        )
        return cursor.fetchall()
    
    def get_user_favorites(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT * FROM favorites WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        )
        return cursor.fetchall()
    
    def close(self):
        self.conn.close()