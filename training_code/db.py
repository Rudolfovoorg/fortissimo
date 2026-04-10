print("we are at db")
from sqlalchemy import select,insert,delete
from sqlalchemy import create_engine
from sqlalchemy import ForeignKey
from sqlalchemy import String
import select as sl
import numpy as np
import threading

from VegaHPCConnection import VegaHPCConnection
from sqlalchemy.orm import sessionmaker 
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from typing import List
import datetime
import os 
from typing import Optional
from Models import Base, EnergyData,Clients,Predictions,LoadPrediction,LoadPrediction1h,Measurements,PVProductionPrediction
import pandas as pd
db_params = {
    'host': '_HOST_',
    'database': '_DATABASE_',
    'user': '_USER_',
    'password': '_PASSWORD_',
    'port': '5432' 
}
db_params_test = {
    'host': '_HOST_',
    'database': '_DATABASE_',
    'user': '_USER_',
    'password': '_PASSWORD_',
    'port': '5432' 
}
env = "production"
if "CONNECTION_STRING" in os.environ and os.environ["CONNECTION_STRING"]:
    if "ENVIRONMENT" in os.environ and os.environ["ENVIRONMENT"]:
        env = os.environ["ENVIRONMENT"]


if env =="development":
    db_params= db_params_test

else:
    print("production")
    #connection_string = os.environ["CONNECTION_STRING"]

        

connection_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
connection_string2 = f"postgresql://{db_params_test['user']}:{db_params_test['password']}@{db_params_test['host']}:{db_params_test['port']}/{db_params_test['database']}"

engine = create_engine(
    connection_string,
    pool_size=10,         # number of concurrent DB connections
    max_overflow=5,       # extra if pool is exhausted
    pool_pre_ping=True,   # recycle stale connections
    pool_recycle=1800     # refresh connections every 30 minutes
)

engine2 = create_engine(
    connection_string2,
    pool_size=10,         # number of concurrent DB connections
    max_overflow=5,       # extra if pool is exhausted
    pool_pre_ping=True,   # recycle stale connections
    pool_recycle=1800     # refresh connections every 30 minutes
)


def listen_for_job_completion(hostname,username,key_file):
    conn = engine.raw_connection()
    conn.set_isolation_level(0) 
    cursor = conn.cursor()
    cursor.execute("LISTEN job_completed;")
    cursor.execute("LISTEN job_failed;")
    print("Listening for job_completed or job failed notifications...")
    i=0
    while True:
        print(f"alive{i}")
        i +=1
        if sl.select([conn], [], [], 10) == ([], [], []):
            continue
        conn.poll()
        while conn.notifies:
            print(f"alive and dead inside")
            notify = conn.notifies.pop()
            job_id = notify.payload
            if notify.channel == 'job_completed':
                print(f"Job {job_id} completed. Initiating file transfer...")
                with VegaHPCConnection(hostname, username) as vega:
                    vega.connect_with_key_and_totp(key_file=key_file)
                    vega.transfer_file_from_HPC()
            elif notify.channel == 'job_failed':    
                print(f" Job {job_id} failed. Consider retrying, alerting, or logging.")
            conn.close()
            return

def PreProcessAndInsert(predicted_values,start,clientId:int):

    try:
        pred_times = pd.date_range(start=start + datetime.timedelta(minutes=15),
                                    periods=4, freq='15min')
        predicted_values_flat = np.asarray(predicted_values).flatten()
        results=[]
            #results
        for t, val in zip(pred_times, predicted_values_flat):
            results.append({'Time': t, 'ClientId': int(clientId), 'LoadPower': val})
        predictions_df = pd.DataFrame(results)
        insert_load_predictions1h(predictions_df)
        consFinishedTime = datetime.datetime.now()
        print(f"[{consFinishedTime}] predictions for {clientId} succesfully inserted")
    except Exception as ex:
        print(f"Error on 1h predict insert {ex}" )

def getPVRelevantData(ClientId):


    query1 = f'SELECT * FROM "Measurements" WHERE "ClientId"={ClientId};'
    query2 = f'''SELECT * FROM "WeatherForecast" WHERE "Time" >= '2025-08-20 00:00:00' AND "ClientId"={ClientId} ;'''
    query3 = f'''SELECT "TimeStampMeasured", "PowerToPv" FROM "EnergyData" WHERE "TimeStampMeasured" >= '2025-08-20 00:00:00' AND "ClientId"={ClientId} ;'''
    try:
        with engine.connect() as conn:
            measurements = pd.read_sql(query1, conn)  # Read into Pandas DataFrames
            measurements.to_csv("measurements.csv", index=False)  # Save to CSV
            print(measurements["Time"].max())
            print('Data saved to measurements.csv!')
            serving_data = pd.read_sql(query2, conn)  # Read into Pandas DataFrame
            serving_data.to_csv("serving_data.csv", index=False)  # Save to CSV
            print(serving_data["Time"].max())
            print('Weather Forecasts data are saved to serving_data.csv!')
            energy_Measurements = pd.read_sql(query3,conn)

            return measurements.reset_index(drop=True),serving_data.reset_index(drop=True),energy_Measurements.reset_index(drop=True)
    except Exception as e:
        print("Query failed:", e)
    
def GetClientDataInTimeframe(clientId:int, start:datetime,end:datetime):
    with Session(engine) as session:
        try:
            stmnt = select(EnergyData.TimeStampMeasured,EnergyData.PowerFromLoad).where(EnergyData.ClientId==clientId, EnergyData.TimeStampMeasured > start, EnergyData.TimeStampMeasured <= end ).order_by(EnergyData.TimeStampMeasured)
            df = pd.read_sql(stmnt,engine)
            return df
        except Exception as ex:
            print(f"Error retrieving timeframe data for client id ={clientId} : {ex}")
            raise
        finally:
            session.close()

def GetClientDataInTimeframe2(clientId:int, start:datetime,end:datetime):
    with Session(engine) as session:
        try:
            stmt = (
                select(EnergyData)
                .where(
                    EnergyData.ClientId == clientId,
                    EnergyData.TimeStampMeasured > start,
                    EnergyData.TimeStampMeasured <= end,
                )
                .order_by(EnergyData.TimeStampMeasured)
            )

            df = pd.read_sql(stmt, engine)
            
            return df
        except Exception as ex:
            print(f"Error retrieving timeframe data for client id ={clientId} : {ex}")
            raise
        finally:
            session.close()
def GetMeasurementsInTimeFrame(clientId:int, start:datetime,end:datetime):
    with Session(engine) as session:
        try:
            stmnt= select(Measurements).where(Measurements.ClientId==clientId, Measurements.Time > start, Measurements.Time <= end ).order_by(Measurements.Time)
            df = pd.read_sql(stmnt,engine)
            return df
        except Exception as ex:
            print(f"Error retrieving timeframe data for client id ={clientId} : {ex}")
            raise
        finally:
            session.close()
def GetEnergyData(clientId:int):
    with Session(engine) as session:
        try:
            stmnt = select(Clients).join(Clients.EnergyData).where(EnergyData.ClientId==clientId)
            df = pd.read_sql(stmnt,engine)
            return df
        except Exception as ex:
            print(f"Error retrieving data: {ex} for Client with id = {clientId}")
            raise
        finally:
            session.close()

def GetClients():
    with Session(engine) as session:
        try:
            stmnt = select(Clients)
            df = pd.read_sql(stmnt,engine) 
            return df
        except Exception as ex:
            print(f"Error retrieving data: {ex}")
            raise
        finally:
            session.close()



def GetData():
    with Session(engine) as session:
        try:
            stmnt = select(EnergyData)
            df = pd.read_sql(stmnt,engine)
            return df
        except Exception as ex:
            print(f"Error retrieving data: {ex}")
            raise



def insert_load_predictions1h(predictions:pd.DataFrame):
    with Session(engine) as session:
        try:
            numOfInserts=predictions.to_sql("LoadPredictions1h",engine,if_exists='append', index=False, method='multi', chunksize=100)
            print(f"inserted {numOfInserts} to database")
            return numOfInserts
        except Exception as ex:
            raise
        finally:
            session.close()
        
def delete_predictions_from(time_now,client_id):
    with Session(engine) as session:
        try:
                # Delete all predictions where the prediction time is greater than time_now
                stmt = delete(LoadPrediction).where((LoadPrediction.Time >= time_now) & (LoadPrediction.ClientId == client_id))
                result = session.execute(stmt)
                
                # Commit the transaction
                session.commit()
                
                print(f"Deleted {result.rowcount} future predictions after {time_now}")
                return result.rowcount
                
        except Exception as e:
            # Rollback in case of error
            session.rollback()
            print(f"Error deleting future predictions: {e}")
            raise e
        
def delete_PVPower_predictions_from(time_now,client_id):
    with Session(engine) as session:
        try:
            # Delete all predictions where the prediction time is greater than time_now
            stmt = delete(PVProductionPrediction).where((PVProductionPrediction.Time >= time_now) & (PVProductionPrediction.ClientId == client_id))
            result = session.execute(stmt)
            
            # Commit the transaction
            session.commit()
            
            print(f"Deleted {result.rowcount} future predictions after {time_now}")
            return result.rowcount
                
        except Exception as e:
            # Rollback in case of error
            session.rollback()
            print(f"Error deleting future predictions: {e}")
            raise e
        
def delete_predictions1h_from(time_now,client_id):
    with Session(engine) as session:
        try:
            # Delete all predictions where the prediction time is greater than time_now
            stmt = delete(LoadPrediction1h).where((LoadPrediction1h.Time > time_now) & (LoadPrediction1h.ClientId == client_id))
            result = session.execute(stmt)
            
            # Commit the transaction
            session.commit()
            
            print(f"Deleted {result.rowcount} future predictions after {time_now}")
            return result.rowcount
                
        except Exception as e:
            # Rollback in case of error
            session.rollback()
            print(f"Error deleting future predictions: {e}")
            raise e
        
def delete_predictions1h_from(time_now,client_id):
    with Session(engine) as session:
        try:
            # Delete all predictions where the prediction time is greater than time_now
            stmt = delete(LoadPrediction1h).where((LoadPrediction1h.Time > time_now) & (LoadPrediction1h.ClientId == client_id))
            result = session.execute(stmt)
            
            # Commit the transaction
            session.commit()
            
            print(f"Deleted {result.rowcount} future predictions after {time_now}")
            return result.rowcount
                
        except Exception as e:
            # Rollback in case of error
            session.rollback()
            print(f"Error deleting future predictions: {e}")
            raise e
        
def insert_load_predictions(predictions:pd.DataFrame,clientId):
    with Session(engine) as session:
        try:
            predictions["ClientId"]=clientId
            predictions["Time"] = pd.to_datetime(predictions["Time"])
            predictions=predictions.rename(columns={'lstm_predictions': 'LSTMLoadPower'})
            predictions=predictions.rename(columns={'hybrid_predictions': 'HybridLoadPower'})
            numOfInserts=predictions.to_sql("LoadPredictions",engine,if_exists='append', index=False, method='multi', chunksize=100)
            print(f"inserted {numOfInserts} to database")
            return numOfInserts
        except Exception as ex:
            raise
        finally:
            session.close()
#!!!!!!!
# still have to rename column
def insert_PVPower_predictions(predictions:pd.DataFrame,clientId):
    with Session(engine) as session:
        try:
            #predictions.reset_index(drop=True)
            predictions=predictions[["Time","SolarPower_pred_lstm"]].copy()
            predictions["ClientId"]=clientId
            predictions=predictions.rename(columns={'SolarPower_pred_lstm': 'PVProductionPower'})
            numOfInserts=predictions.to_sql("PVProductionPrediction",engine,if_exists='append', index=False, method='multi', chunksize=100)
            print(f"inserted {numOfInserts} to database")
            return numOfInserts
        except Exception as ex:
            raise
        finally:
            session.close()

def GetDataInTimeframe(start:datetime,end:datetime):
    with Session(engine) as session:
        try:
            stmnt = select(EnergyData).where(EnergyData.TimeStampMeasured > start, EnergyData.TimeStampMeasured <= end ).order_by(EnergyData.TimeStampMeasured)
            df= pd.read_sql(stmnt,engine)
            return df
        except Exception as ex:
            print(f"Error retrieving data: {ex}")
            raise
        finally:
            session.close()

def InsertClient(client:Clients):
    with Session(engine) as session:
        try:
            session.add(client)
            session.commit()
            print("insert succesful")
        except Exception as ex:
            print(f"Error inserting Client: {ex}")
            raise
        finally:
            session.close()


