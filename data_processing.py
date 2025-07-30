"""
data_processing.py - Procesamiento de datos GPS
Adaptado para estructura de carpetas anidadas y múltiples vehículos
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import config

SPEED_EXCESS_RE = re.compile(
    r"Exceso de Velocidad:\s*([0-9]+(?:\.[0-9]+)?)\s*km/h\s+en\s+zona\s+de"
    r"\s*([0-9]+(?:\.[0-9]+)?)\s*km/h",
    re.IGNORECASE
)
MINUTES_RE = re.compile(r"durante\s*(\d+)\s*minutos?", re.IGNORECASE)

def clean_text(text):
    """Limpia texto de paréntesis y corchetes"""
    return re.sub(r"[\\[\\]\\(\\)]", "", str(text)).strip()

def load_excel_file(filepath):
    """Carga y preprocesa un archivo Excel de GPS"""
    df = pd.read_excel(filepath, header=13, engine="openpyxl")
    df.columns = df.columns.str.strip()
    df["Hora"] = pd.to_datetime(df["Hora"], format="%d/%m/%Y %H:%M:%S")
    df = df.sort_values("Hora").reset_index(drop=True)
    df["Texto"] = df["Evento"].fillna("") + " " + df["Flags"].fillna("")
    df["Texto"] = df["Texto"].apply(clean_text)
    df["Vel_kmh"] = pd.to_numeric(df["Velocidad (km/h)"], errors="coerce").fillna(0)
    df["dt_sec"] = df["Hora"].diff().dt.total_seconds().fillna(0)
    
    if "Latitud" in df.columns and "Longitud" in df.columns:
        df["Latitud"] = pd.to_numeric(df["Latitud"], errors="coerce")
        df["Longitud"] = pd.to_numeric(df["Longitud"], errors="coerce")
    
    return df

def discover_vehicles(base_dir=None):
    """
    Descubre todos los vehículos disponibles en el directorio
    Retorna dict con info de cada vehículo
    """
    if base_dir is None:
        base_dir = config.DATA_DIR
    
    vehicles = {}
    vehicle_pattern = re.compile(config.VEHICLE_FOLDER_PATTERN)
    
    for folder in Path(base_dir).iterdir():
        if folder.is_dir():
            match = vehicle_pattern.match(folder.name)
            if match:
                vehicle_id = match.group(1)
                date_start = datetime.strptime(match.group(2), config.DATE_FORMAT_FILE).date()
                date_end = datetime.strptime(match.group(3), config.DATE_FORMAT_FILE).date()
                
                if vehicle_id not in vehicles:
                    vehicles[vehicle_id] = {
                        'folders': [],
                        'date_range': [date_start, date_end],
                        'total_days': 0
                    }
                else:
                    vehicles[vehicle_id]['date_range'][0] = min(vehicles[vehicle_id]['date_range'][0], date_start)
                    vehicles[vehicle_id]['date_range'][1] = max(vehicles[vehicle_id]['date_range'][1], date_end)
                
                vehicles[vehicle_id]['folders'].append(folder)
    
    for vehicle_id in vehicles:
        vehicles[vehicle_id]['total_days'] = count_available_days(vehicle_id, base_dir)
    
    return vehicles

def count_available_days(vehicle_id, base_dir=None):
    """Cuenta cuántos días de datos hay disponibles para un vehículo"""
    if base_dir is None:
        base_dir = config.DATA_DIR
    
    daily_pattern = re.compile(config.DAILY_FOLDER_PATTERN)
    available_dates = set()
    
    for folder in Path(base_dir).iterdir():
        if folder.is_dir() and vehicle_id in folder.name:
            for daily_folder in folder.iterdir():
                if daily_folder.is_dir():
                    match = daily_pattern.match(daily_folder.name)
                    if match and match.group(2) == vehicle_id:
                        date_str = match.group(1)
                        available_dates.add(date_str)
    
    return len(available_dates)

def find_excel_files_for_vehicle(vehicle_id, start_date=None, end_date=None, base_dir=None):
    """
    Encuentra todos los archivos Excel para un vehículo en un rango de fechas
    """
    if base_dir is None:
        base_dir = config.DATA_DIR
    
    excel_files = []
    daily_pattern = re.compile(config.DAILY_FOLDER_PATTERN)
    excel_pattern = re.compile(config.EXCEL_FILE_PATTERN)
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, config.DATE_FORMAT_FILE).date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, config.DATE_FORMAT_FILE).date()
    
    for folder in Path(base_dir).iterdir():
        if folder.is_dir() and vehicle_id in folder.name:
            for daily_folder in folder.iterdir():
                if daily_folder.is_dir():
                    match = daily_pattern.match(daily_folder.name)
                    if match and match.group(2) == vehicle_id:
                        file_date = datetime.strptime(match.group(1), config.DATE_FORMAT_FILE).date()
                        
                        if start_date and file_date < start_date:
                            continue
                        if end_date and file_date > end_date:
                            continue
                        
                        for file in daily_folder.iterdir():
                            if file.is_file() and excel_pattern.match(file.name):
                                excel_files.append({
                                    'filepath': file,
                                    'date': file_date,
                                    'vehicle': vehicle_id,
                                    'filename': file.name
                                })
                                break
    
    excel_files.sort(key=lambda x: x['date'])
    return excel_files

def detect_speed_excesses(df):
    """Detecta excesos de velocidad"""
    df["Evento_clean"] = df["Evento"].fillna("").apply(clean_text)
    df["is_exceso"] = df["Evento_clean"].apply(lambda t: bool(SPEED_EXCESS_RE.search(t)))
    
    df["run_id"] = (df["is_exceso"] != df["is_exceso"].shift(1)).cumsum()
    
    durations = []
    excesses_info = []
    
    for run_id, group in df.groupby("run_id"):
        if not group["is_exceso"].iloc[0]:
            continue
            
        start_ts = group["Hora"].iloc[0]
        end_idx = group.index.max()
        
        next_idx = end_idx + 1 if end_idx + 1 in df.index else None
        if next_idx is not None:
            end_ts = df.at[next_idx, "Hora"]
            raw_sec = (end_ts - start_ts).total_seconds()
        else:
            raw_sec = group["dt_sec"].sum()
            
        mins_match = group["Evento_clean"].str.extract(MINUTES_RE)
        if mins_match[0].notna().any():
            mins = int(mins_match[0].dropna().astype(int).max())
            raw_sec = mins * 60 + raw_sec
            
        durations.append(raw_sec)
        
        excesses_info.append({
            "start_time": start_ts,
            "duration_sec": raw_sec,
            "start_idx": group.index[0],
            "end_idx": end_idx
        })
    
    return durations, excesses_info

def calculate_driving_metrics(df, durations, excesses_info):
    """Calcula métricas de conducción agregadas"""
    fecha = df["Hora"].dt.date.iloc[0]
    
    mov = df[df["Vel_kmh"] > 1]["Vel_kmh"]
    vel_media_mov = mov.mean() if not mov.empty else 0
    vel_max = df["Vel_kmh"].max()
    
    distancia_km = (df["Vel_kmh"] * df["dt_sec"] / 3600).sum()
    
    num_excesos = len(durations)
    dur_exceso_tot_sec = sum(durations)
    excesos_por_km = num_excesos / distancia_km if distancia_km > 0 else 0
    
    df["tramo"] = df["Hora"].dt.floor("T")
    df["delta_v"] = df["Vel_kmh"].diff().fillna(0)
    df["acc_kmhps"] = (df["delta_v"] / df["dt_sec"].replace(0, np.nan)).fillna(0)
    
    win = df.groupby("tramo")
    harsh_accel_w = (win["acc_kmhps"].max() > 10).sum()
    harsh_brake_w = (win["acc_kmhps"].min() < -10).sum()
    
    df["hora"] = df["Hora"].dt.hour
    df["franja"] = pd.cut(
        df["hora"], 
        bins=[0, 6, 12, 18, 24], 
        labels=["Madrugada", "Mañana", "Tarde", "Noche"]
    )
    
    excesos_por_franja = {}
    for exc in excesses_info:
        hora = df.loc[exc["start_idx"], "hora"]
        franja = df.loc[exc["start_idx"], "franja"]
        excesos_por_franja[franja] = excesos_por_franja.get(franja, 0) + 1
    
    return {
        "fecha": fecha,
        "vel_media_mov": vel_media_mov,
        "vel_max": vel_max,
        "distancia_km": distancia_km,
        "num_excesos": num_excesos,
        "dur_exceso_tot_sec": dur_exceso_tot_sec,
        "excesos_por_km": excesos_por_km,
        "harsh_accel_windows": int(harsh_accel_w),
        "harsh_brake_windows": int(harsh_brake_w),
        "excesos_por_franja": excesos_por_franja,
        "tiempo_conduccion_min": mov.count() * df["dt_sec"].mean() / 60 if not mov.empty else 0
    }

def process_vehicle_files(vehicle_id, start_date=None, end_date=None, base_dir=None):
    """
    Procesa todos los archivos de un vehículo en un rango de fechas
    """
    if base_dir is None:
        base_dir = config.DATA_DIR
    
    excel_files = find_excel_files_for_vehicle(vehicle_id, start_date, end_date, base_dir)
    
    if not excel_files:
        return pd.DataFrame(), {}
    
    results = []
    files_data = {}
    
    for file_info in excel_files:
        try:
            filepath = file_info['filepath']
            
            df = load_excel_file(filepath)
            durations, excesses_info = detect_speed_excesses(df)
            metrics = calculate_driving_metrics(df, durations, excesses_info)
            
            metrics['vehicle_id'] = vehicle_id
            
            results.append(metrics)
            files_data[metrics["fecha"]] = {
                "df": df,
                "excesses": excesses_info,
                "durations": durations,
                "filename": file_info['filename'],
                "vehicle_id": vehicle_id
            }
            
        except Exception as e:
            print(f"Error procesando {file_info['filename']}: {e}")
            continue
    
    return pd.DataFrame(results), files_data

def process_all_files(directory=None):
    """
    Función legacy para compatibilidad - procesa archivos en formato anterior
    """
    if directory is None:
        directory = config.DATA_DIR
        
    results = []
    files_data = {}
    
    excel_files = sorted(Path(directory).glob("*.xlsx"))
    
    if not excel_files:
        return pd.DataFrame(), {}
    
    for filepath in excel_files:
        try:
            df = load_excel_file(filepath)
            durations, excesses_info = detect_speed_excesses(df)
            metrics = calculate_driving_metrics(df, durations, excesses_info)
            
            results.append(metrics)
            files_data[metrics["fecha"]] = {
                "df": df,
                "excesses": excesses_info,
                "durations": durations,
                "filename": filepath.name
            }
            
        except Exception as e:
            print(f"Error procesando {filepath.name}: {e}")
            continue
    
    return pd.DataFrame(results), files_data

def calculate_real_distance(df, max_jump_km=1.0):
    """Calcula distancia real usando GPS"""
    if "Latitud" not in df.columns or "Longitud" not in df.columns:
        return None, []
        
    df_gps = df.dropna(subset=["Latitud", "Longitud"]).copy()
    
    if len(df_gps) < 2:
        return None, []
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    distances = []
    valid_coords = []
    
    prev_lat = df_gps.iloc[0]["Latitud"]
    prev_lon = df_gps.iloc[0]["Longitud"]
    valid_coords.append((prev_lat, prev_lon))
    
    for idx in range(1, len(df_gps)):
        curr_lat = df_gps.iloc[idx]["Latitud"]
        curr_lon = df_gps.iloc[idx]["Longitud"]
        
        dist = haversine(prev_lat, prev_lon, curr_lat, curr_lon)
        
        if dist <= max_jump_km:
            distances.append(dist)
            valid_coords.append((curr_lat, curr_lon))
            prev_lat, prev_lon = curr_lat, curr_lon
    
    return sum(distances), valid_coords