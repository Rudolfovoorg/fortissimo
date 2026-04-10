import psycopg2
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Connection parameters
# Database connection details
DB_HOST = "_HOST_"
DB_PORT = "5432"
DB_NAME = "_DATABASE_"
DB_USER = "_USER_"
DB_PASSWORD = "_PASSWORD_"

# Connect to PostgreSQL
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

query1 = 'SELECT * FROM "Measurements";'


try:
    df = pd.read_sql(query1, conn)  # Read into Pandas DataFrame
    df.to_csv("_WEATHER_OUTPUT_PATH_", index=False)  # Save to CSV
    print('Data saved to weather.csv!')

except Exception as e:
    print("Query failed:", e)

# Close the connection
conn.close()
print("Connection closed.")
