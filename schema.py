import sqlite3
import os

# Find the file in the exact folder the script is sitting in
db_path = os.path.join(os.path.dirname(__file__), "ssd_analysis")
schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

with sqlite3.connect(db_path) as conn:
    with open(schema_path, "w") as f:
        for row in conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"):
            f.write(row[0] + ";\n\n")

print("Done creating schema.sql file!")

