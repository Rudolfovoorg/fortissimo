import psycopg2
import pandas as pd
import os
import warnings;
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable*")

# Connection parameters

DB_HOST = "_HOST_"
DB_PORT = "5432"
DB_NAME = "_DATABASE_"
DB_USER = "_USER_"
DB_PASSWORD = "_PASSWORD_"

# Connect
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    print("Successfully connected to the database!")
except Exception as e:
    print("Connection failed:", e)
    exit()


query = 'SELECT * FROM "EnergyData";'

os.chdir("_WORKING_DIR_")


try:
    df = pd.read_sql(query, conn)
    df.to_csv("load_calculated.csv", index=False)
    print('Data saved to load_calculated.csv!')
except Exception as e:
    print("Query failed:", e)

# Close the connection
conn.close()
print("Connection closed.")
