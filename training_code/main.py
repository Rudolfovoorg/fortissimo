print("we are at main")
from Models import Clients
from sqlalchemy.orm import sessionmaker
from VegaHPCConnection import VegaHPCConnection
print("Importing database...")
import db
import pandas as pd
print("Imported database...")
from datetime import datetime,timedelta, timezone
import pytz
import time
import threading
import numpy as np
import requests
import os
print("Importing dl...")
import predictions_interval as dl
print("imported dl...")
import ml_functions as ml
from pv_data_validation import serving_data,feature_processing
from pv_predict import train_pv_model
import sys 
from loadPredictV2 import prep_data, train_model 
from pathlib import Path

# data = db.GetClientData(7)
    #check month
    # mesec 11 - 2 -> v
    # mesec 3 - 10 -> n
    # check day  if dela prost dan 
    # print(dataPoint.EnergyFromGrid)
def main():
    print(datetime.now())
    startTime = datetime.now()
    solarFinishedTime = datetime(1, 1, 1, 0, 0)
    consFinishedTime = datetime(1, 1, 1, 0, 0)
    consTrainFinishedTime = datetime(1, 1, 1, 0, 0)
    toggle = True
    print("Main Start")
    
    env = "development"

    # read env variables from docker
    if "CONNECTION_STRING" in os.environ and os.environ["CONNECTION_STRING"]:
        conn_str = os.environ["CONNECTION_STRING"]
    if "ENVIRONMENT" in os.environ and os.environ["ENVIRONMENT"]:
        env = os.environ["ENVIRONMENT"]

    if env == "development":
        # CONNECTIONS STRING FOR TESTING; PRODUCTION GETS PRODUCTION CONNECTION STRING FROM ENV VARIABLES
        # conn_str = read_connection_string()
        print("dev")
    env="production"
    production = True if env == "production" else False
    print(env)    
    toggle = True#print(startTime, production, solarFinishedTime, consFinishedTime, consTrainFinishedTime)
    print("Init OK")
    
    
    # this runs immediately when starting programm in development
    if not production:


        
        print("in dev mode")
      #production    
    elif  production:

        print("MAIN LOOP STARTING")
        hostname= "_HOST_"
        username= "_USER_"
        ssh_key = r"_SSH_KEY_PATH_"
        submit_run_training="./submit_train36ml.sh" 
        # MAIN LOOP/timer,
        hourlyPredictInterval = 3600  # Run health check every hour (3600 seconds)
        last_hpredict_check = datetime.now()
        start_date=datetime(2025, 3, 12) 
        


        while production:
            try:
                current_time = datetime.now()
                current_time_striped = current_time.replace(second=0, microsecond=0)

                print(current_time)
                if current_time.minute == 45 and current_time.second < 30:
                    try:
                        clients= db.GetClients()
                        
                        for client in clients.itertuples():
                            db.delete_predictions1h_from(current_time_striped,client.Id)

                            energyData= db.GetClientDataInTimeframe(client.Id,start_date,current_time_striped)
                            energyData=energyData[(energyData["PowerFromLoad"]<  150000) & (energyData["PowerFromLoad"]>=3000)]
                            energyData.to_csv("enrgyTest.csv")

                            measurements= db.GetMeasurementsInTimeFrame(client.Id,start_date,current_time_striped)
                            measurements.to_csv("measurementsTEst.csv")

                            agregated_weather_data= ml.data_agregate_weather(energy_data_cond=energyData,measurements_df=measurements)
                            agregated_weather_data.to_csv("agregatedTest.csv")

                            predictions= ml.make_hourly_prediction(df_weather_agregated=agregated_weather_data,current_time=current_time_striped)
                            print(predictions)

                            db.PreProcessAndInsert(predicted_values=predictions,clientId=client.Id,start=current_time_striped)
                            
                        print(f"[{current_time}] hourly predictor completed successfully")
                        last_hpredict_check = current_time
                        time.sleep(30)
                    except Exception as health_error:
                        print(f"hourly predictor failed: {str(health_error)}")
                
                if  False:#current_time.day % 7 == 0 and current_time.hour == 1 and current_time.minute == 43 and current_time.second < 30 and toggle:
                    clients= db.GetClients()

                    #VEGA full job submissioin and 
                    with VegaHPCConnection(hostname, username) as vega:
                        vega.connect_with_key_and_totp(key_file=ssh_key)
                        vega.submit_comand(submit_run_training)
                        #vega.transfer_file_from_HPC()
                        listener_thread = threading.Thread(
                        target=db.listen_for_job_completion,
                        args=(hostname,username,ssh_key),
                        daemon=True  # Kills with main process
                        )
                        listener_thread.start()

                        # Main thread can continue or wait
                        #listener_thread.join() # wait on thread to finish thenrun loop forward

                    for client in clients.itertuples():
                        EnergyData = db.GetClientDataInTimeframe(client.Id,start_date,current_time)
                        #ml.train_hourly_model(client.Id,EnergyData)
                        #print(EnergyData)
                    consTrainFinishedTime = datetime.now()
                    time.sleep(30)
                    
                elif current_time.hour == 0 and current_time.minute ==1 and current_time.second < 30 and toggle:
                    if True:#check_consumption_model():
                        clients= db.GetClients()
                        local_tz = pytz.timezone("Europe/Ljubljana")
                        today_midnight = datetime.now(local_tz).replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )

                        for client in clients.itertuples():
                            # if (False): # testing predicition pipeline
                            #db.delete_PVPower_predictions_from(today_midnight-timedelta(days=7),client.Id)
                            dl.file_cleanup()
                            measurements , serving_data1,energy_data =db.getPVRelevantData(client.Id)
                            train_test_data= ml.pv_data_agregation(measurements,energy_data)
                            train_test_data.dropna(inplace=True)
                            print(train_test_data.isna().sum())
                            df1, df_selected= feature_processing(train_test_data)
                            df_weather_expanded = serving_data(df1,serving_data1) 
                            predicted_df= train_pv_model(df_selected,df_weather_expanded,train_test_data,client.Id)
                            predicted_df =  predicted_df[predicted_df["Time"] >=today_midnight-timedelta(days=7)]
                            #predicted_df.to_csv("testingLatestŠoncComent.csv")
                            #db.insert_PVPower_predictions(predicted_df,client.Id)
                            url = "_DISTRIBUTE_API_URL_"
                            def  run_predicitionV2_pipeline(): 
                                try:
                                    
                                    headers = {
                                        "X-API-Key": "_API_KEY_"
                                    } 
                                    response = requests.get(url, headers=headers)
                                    response.raise_for_status()
                                    print("Fetched:", response.status_code)

                                    print("prep_energy start")
                                    energy_df = prep_data.prep_energy(client.Id)
                                    print("prep_energy done")

                                    print("prep_weather start")
                                    weather_df = prep_data.prep_weather(client.Id)
                                    print("prep_weather done")

                                    print("prep_features start")
                                    prepared_df = prep_data.prep_features(energy_df, weather_df)
                                    print("prep_features done")

                                    print("train_model start")
                                    train_model.train_model(client.Id, prepared_df)
                                    print("train_model done")

                                    print("forecast start")
                                    df = prep_data.forecast_next_day(client.Id, prepared_df)
                                    print("forecast done")
                                    return df
                                except requests.RequestException as e:
                                    print("GET request failed:", e)
                                    return
                            dayli_predicitions= run_predicitionV2_pipeline()

                            # consolidation from previous insert logic
                            dayli_predicitions=dayli_predicitions.rename(columns={'LoadEnergyForecast_15': 'LSTMLoadPower'})
                            dayli_predicitions=dayli_predicitions.rename(columns={'Timestamp': 'Time'})
                            dayli_predicitions["hybrid_predictions"]=0
                            dayli_predicitions['Time'] = (  
                                pd.to_datetime(dayli_predicitions['Time'], utc=True)
                                .dt.tz_convert('Europe/Ljubljana')
                            )
                            # if True:
                            db.delete_predictions_from(pd.to_datetime(dayli_predicitions.iloc[0]["Time"]),client.Id)
                            db.insert_load_predictions(dayli_predicitions,client.Id) # using the test db !!!!


                            # ///////// OLD MODEL ////////////////////
                            # energyData= db.GetClientDataInTimeframe(client.Id,start_date,today_midnight)
                            # #energyData.to_csv("test1.csv")
                            # energyData=energyData[(energyData["PowerFromLoad"]<  150000) & (energyData["PowerFromLoad"]>=3000)]
                            # print(today_midnight)
                            
                            # agregated_df= ml.data_agregation(energyData)
                            # print(agregated_df)
                            # dl.train_model_consumption_lstm(agregated_df,client.Id)
                            # pred_df= dl.make_predictions_consumption_lstm(agregated_df,client.Id)
                            # dl.train_model_consumption_hybrid(agregated_df,pred_df,client.Id)
                            # predictions_hybrid= dl.make_predictions_consumption_hybrid(pred_df,client.Id)
                            # db.insert_load_predictions(predictions_hybrid,client.Id)
                            # ///////// OLD MODEL END ////////////////////
                            print("procedure complete")
                    time.sleep(30)

                else:
                    if current_time.second <= 14:
                        print("log")
                        #toggle = log(startTime, production, solarFinishedTime, consFinishedTime, consTrainFinishedTime)
                    time.sleep(15)  # Sleep for 15s
                    
            except Exception as e:
                print(f"An unexpected error occurred: {str(e)}")
                time.sleep(30)
                
        print("Main End")


if __name__ == "__main__":
    main()
