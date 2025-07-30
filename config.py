"""
config.py - Configuración centralizada del sistema MODIFICADO
Adaptado para estructura de carpetas anidadas y múltiples vehículos
"""
import os
from pathlib import Path

# Configuración de rutas - MODIFICADO para estructura de carpetas
BASE_DIR = Path(__file__).resolve().parent  # Directorio donde está este archivo
DATA_DIR = BASE_DIR  # Los datos están en el mismo directorio base

# Configuración de API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "tu-api-key-aqui")

# Configuración de procesamiento
EXCEL_HEADER_ROW = 13
DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S"

# NUEVO: Configuración para estructura de carpetas
VEHICLE_FOLDER_PATTERN = r"^([A-Z0-9]{6})_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$"
DAILY_FOLDER_PATTERN = r"^(\d{4}-\d{2}-\d{2})_([A-Z0-9]{6})$"
EXCEL_FILE_PATTERN = r"^(\d{4}-\d{2}-\d{2})_([A-Z0-9]{6})\.xlsx$"

# Umbrales de análisis (SIN CAMBIOS)
SPEED_THRESHOLD_MOVING = 1
HARSH_ACCEL_THRESHOLD = 10
HARSH_BRAKE_THRESHOLD = -10

# Configuración de clustering (SIN CAMBIOS)
N_CLUSTERS = 3
CLUSTER_NAMES = {
    0: "Conducción Segura",
    1: "Conducción Moderada", 
    2: "Conducción Arriesgada"
}

# Configuración de LLM (SIN CAMBIOS)
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500

# Colores para visualizaciones (SIN CAMBIOS)
CLUSTER_COLORS = {
    0: "#2ecc71",
    1: "#f39c12",
    2: "#e74c3c"
}

# Configuración de Streamlit (SIN CAMBIOS)
PAGE_TITLE = "DriveTech - Análisis de Conducción"
PAGE_ICON = "🚗"
LAYOUT = "wide"

# NUEVO: Configuración para interfaz de vehículos
DEFAULT_VEHICLE = "TWJL30"  # Vehículo por defecto
DATE_FORMAT_DISPLAY = "%d/%m/%Y"  # Formato para mostrar fechas
DATE_FORMAT_FILE = "%Y-%m-%d"     # Formato en nombres de archivo

# NUEVO: Configuración para score mejorado
DEFAULT_SPEED_THRESHOLD = 85  # km/h - Umbral de velocidad de empresa
MAX_GPS_GAP_SECONDS = 120     # Máximo gap en segundos para evitar errores GPS

# NUEVO: Umbrales absolutos para clustering (reemplaza K-Means)
RISK_SCORE_THRESHOLDS = {
    "seguro": 30,      # < 30 = Conducción Segura
    "moderado": 60,    # 30-60 = Conducción Moderada  
    # > 60 = Conducción Arriesgada
}