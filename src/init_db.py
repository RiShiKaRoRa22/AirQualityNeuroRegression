import sqlite3 as sq3

conn=sq3.connect("db/air_quality.db")
cursor= conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS raw_api_data (
    location TEXT,
    date_utc TEXT,
    parameter TEXT,
    value REAL
);
""")

conn.commit()
conn.close()

print("SQLite DB and table created successfully.")