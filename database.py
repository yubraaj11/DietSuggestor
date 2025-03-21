import sqlite3

DB_NAME = "db/fit_foodie.db"

def create_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            age INTEGER NOT NULL,
            weight REAL NOT NULL,
            gender TEXT CHECK(gender IN ('male', 'female')) NOT NULL,
            height INTEGER NOT NULL
        )
    """)

    # Macronutrients table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS macronutrients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            meal_type TEXT NOT NULL,
            recommended_diet TEXT NOT NULL,
            calories_intake REAL NOT NULL,
            protein REAL NOT NULL,
            carbohydrates REAL NOT NULL,
            fats REAL NOT NULL,
            fibre REAL NOT NULL,
            date TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

    """)

    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

create_table()
