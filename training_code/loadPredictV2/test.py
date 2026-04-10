import prep_data
import train_model
import requests
url = "_LOCAL_DISTRIBUTE_API_URL_"
def  run_predicitionV2_pipeline(): 
    try: 
        response = requests.get(url)
        response.raise_for_status()

        energy_df= prep_data.prep_energy(4)
        weather_df= prep_data.prep_weather(4)
        
        prepered_df=prep_data.prep_features(data_energy=energy_df,data_weather=weather_df)
        train_model.train_model(4,prepered_df)
        
        #prep_data.forecast_next_day(4,prepered_df)
        print("test end")
    except requests.RequestException as e:
        print("GET request failed:", e)
        return

if __name__ == "__main__":
    run_predicitionV2_pipeline()
