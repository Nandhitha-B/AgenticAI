import sqlite3

conn = sqlite3.connect("backend/database/app.db")
cursor = conn.cursor()

print("\n--- MODELS TABLE ---")
cursor.execute("SELECT * FROM models")
for row in cursor.fetchall():
    print(row)

print("\n--- IMAGES TABLE ---")
cursor.execute("SELECT * FROM images")
for row in cursor.fetchall():
    print(row)

print("\n--- METRICS TABLE ---")
cursor.execute("SELECT * FROM metrics")
for row in cursor.fetchall():
    print(row)

conn.close()
