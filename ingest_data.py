# ingest_data.py (Updated to process a limited number of files)
import xarray as xr
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
import toml
from ftplib import FTP
import os
import numpy as np

# --- Configuration is unchanged ---
FTP_HOST = "ftp.ifremer.fr"
LOCAL_INDEX_FILE = "ar_index_global_meta.txt"
LOCAL_DATA_DIR = "argo_data"
secrets = toml.load(".streamlit/secrets.toml")
DB_CONNECTION_STRING = secrets["connections"]["postgres"]["url"]

# --- find_float_dir_local and get_var functions are unchanged ---
def find_float_dir_local(float_id):
    # ... (code is the same)
    print(f"Searching for directory for float {float_id} in local meta index...")
    if not os.path.exists(LOCAL_INDEX_FILE):
        print(f"Error: Index file '{LOCAL_INDEX_FILE}' not found. Please run download_index.py first.")
        return None
    with open(LOCAL_INDEX_FILE, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('#'): continue
            if f"/{float_id}/" in line:
                path_to_meta_file = line.split(',')[0]
                directory = os.path.dirname(path_to_meta_file)
                print(f"Found directory: {directory}")
                return directory
    print("Directory not found in index file.")
    return None

def get_var(ds, primary, fallback):
    # ... (code is the same)
    if primary in ds: return ds[primary]
    elif fallback in ds: return ds[fallback]
    raise KeyError(f"Neither '{primary}' nor '{fallback}' found in dataset.")

def process_single_profile_file(local_filepath):
    # ... (code is the same)
    all_profiles_list = []
    try:
        with xr.open_dataset(local_filepath) as ds:
            profile_numbers = ds['CYCLE_NUMBER'].values
            num_profiles_in_file = len(profile_numbers)
            for i in range(num_profiles_in_file):
                profile_data = ds.isel(N_PROF=i)
                data = {
                    'N_PROF': profile_numbers[i],
                    'LATITUDE': profile_data['LATITUDE'].values.item(),
                    'LONGITUDE': profile_data['LONGITUDE'].values.item(),
                    'JULD': profile_data['JULD'].values.item()
                }
                pres_data = get_var(profile_data, 'PRES_ADJUSTED', 'PRES').values
                data['PRES'] = pres_data
                if 'TEMP_ADJUSTED' in profile_data or 'TEMP' in profile_data:
                    data['TEMP'] = get_var(profile_data, 'TEMP_ADJUSTED', 'TEMP').values
                if 'PSAL_ADJUSTED' in profile_data or 'PSAL' in profile_data:
                    data['PSAL'] = get_var(profile_data, 'PSAL_ADJUSTED', 'PSAL').values
                if 'DOXY_ADJUSTED' in profile_data or 'DOXY' in profile_data:
                    data['DOXY'] = get_var(profile_data, 'DOXY_ADJUSTED', 'DOXY').values
                if 'CHLA_ADJUSTED' in profile_data or 'CHLA' in profile_data:
                    data['CHLA'] = get_var(profile_data, 'CHLA_ADJUSTED', 'CHLA').values
                df = pd.DataFrame(data)
                df.dropna(subset=['PRES'], inplace=True)
                all_profiles_list.append(df)
        return pd.concat(all_profiles_list, ignore_index=True) if all_profiles_list else None
    except Exception as e:
        print(f"Could not process file {os.path.basename(local_filepath)}. Error: {e}")
        return None
    finally:
        if os.path.exists(local_filepath):
            os.remove(local_filepath)

def ingest_all_float_data(ftp_host, local_dir, float_id, float_directory):
    """Connects to FTP once, downloads a SUBSET of profiles, and combines them."""
    all_profiles_list = []
    ftp = None
    try:
        ftp = FTP(ftp_host)
        ftp.login()
        target_dir = f"/ifremer/argo/dac/{float_directory}/profiles/"
        print(f"Navigating to FTP directory: {target_dir}")
        ftp.cwd(target_dir)
        
        profile_files = [f.strip() for f in ftp.nlst() if f.strip().endswith('.nc')]

        # --- THIS IS THE NEW LINE TO REDUCE THE FILE COUNT ---
        profile_files = profile_files[:50] # ✂️ Process only the first 50 files
        
        print(f"Found {len(profile_files)} profile files to process (limited to 50).")
        
        for i, pf in enumerate(profile_files):
            local_filepath = os.path.join(local_dir, pf)
            print(f"Processing file {i+1}/{len(profile_files)}: {pf}")
            with open(local_filepath, 'wb') as f:
                ftp.retrbinary(f"RETR {pf}", f.write)
            
            processed_df = process_single_profile_file(local_filepath)
            if processed_df is not None:
                all_profiles_list.append(processed_df)

        if not all_profiles_list: return pd.DataFrame()
        return pd.concat(all_profiles_list, ignore_index=True)
    except Exception as e:
        print(f"An FTP error occurred: {e}")
        return None
    finally:
        if ftp: ftp.quit()

def populate_cloud_database(final_df):
    # ... (code is the same)
    print("--- Starting Database Population ---")
    rename_map = {
        'N_PROF': 'n_prof', 'LATITUDE': 'latitude', 'LONGITUDE': 'longitude', 'JULD': 'timestamp',
        'PRES': 'pressure', 'TEMP': 'temperature', 'PSAL': 'salinity',
        'DOXY': 'doxy_adjusted', 'CHLA': 'chla_adjusted'
    }
    final_df = final_df.rename(columns=rename_map)
    engine = create_engine(DB_CONNECTION_STRING)
    gdf = gpd.GeoDataFrame(
        final_df, geometry=gpd.points_from_xy(final_df.longitude, final_df.latitude), crs="EPSG:4326"
    )
    gdf.columns = [col.lower() for col in gdf.columns]
    gdf.to_postgis('argo_profiles', engine, if_exists='replace', index=False)
    print("--- Cloud Database Population Complete! ---")

if __name__ == '__main__':
    # ... (code is the same)
    FLOAT_TO_INGEST = '1902671'
    directory = find_float_dir_local(FLOAT_TO_INGEST)
    if directory:
        final_dataframe = ingest_all_float_data(FTP_HOST, LOCAL_DATA_DIR, FLOAT_TO_INGEST, directory)
        if final_dataframe is not None and not final_dataframe.empty:
            populate_cloud_database(final_dataframe)