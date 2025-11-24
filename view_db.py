
import sqlite3

DATABASE = 'f1.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def view_database():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables in the database:")
        for table in tables:
            print(table['name'])
        
        for table in tables:
            table_name = table['name']
            print(f"\nData from table: {table_name}")
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            for row in rows:
                print(dict(row))

if __name__ == '__main__':
    view_database()
