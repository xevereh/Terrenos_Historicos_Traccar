"""
debug_excesos.py - Script para debuggear la detección de excesos
Ejecutar este script directamente para ver qué está pasando
"""
import pandas as pd
import re
from pathlib import Path

# Configuración
DATA_DIR = Path(".")  # Directorio actual
EXCEL_HEADER_ROW = 13

# Regex actualizado para capturar AMBOS formatos
# Formato 1: "Exceso de Velocidad: 101 km/h en zona de..."
# Formato 2: "Exceso de Velocidad (95 km/h)"
SPEED_EXCESS_RE = re.compile(
    r"Exceso de Velocidad[:\s]*(?:\()?([0-9]+(?:\.[0-9]+)?)\s*km/h"  # captura velocidad con : o (
    r"(?:.*?zona de\s*([0-9]+(?:\.[0-9]+)?)\s*km/h)?",              # captura límite si existe
    re.IGNORECASE
)

def clean_text(text):
    """Limpia texto de paréntesis y corchetes"""
    return re.sub(r"[\\[\\]\\(\\)]", "", str(text)).strip()

def debug_single_file(filepath):
    """Debug detallado de un archivo"""
    print(f"\n{'='*80}")
    print(f"DEBUGGEANDO ARCHIVO: {filepath.name}")
    print(f"{'='*80}")
    
    # Leer archivo
    df = pd.read_excel(filepath, header=EXCEL_HEADER_ROW, engine="openpyxl")
    df.columns = df.columns.str.strip()
    
    print(f"\nColumnas encontradas: {list(df.columns)}")
    print(f"Total de registros: {len(df)}")
    
    # Verificar si existen las columnas esperadas
    if "Evento" not in df.columns:
        print("⚠️ ADVERTENCIA: No se encontró columna 'Evento'")
    if "Flags" not in df.columns:
        print("⚠️ ADVERTENCIA: No se encontró columna 'Flags'")
    
    # IMPORTANTE: Detectar excesos SOLO en columna Evento
    df["Evento_clean"] = df["Evento"].fillna("").apply(clean_text)
    
    # Buscar cualquier texto que contenga "exceso" EN EVENTO
    print("\n--- BÚSQUEDA DE TEXTOS CON 'EXCESO' EN COLUMNA EVENTO ---")
    exceso_mask = df["Evento_clean"].str.contains("exceso", case=False, na=False)
    textos_con_exceso = df[exceso_mask]["Evento_clean"].unique()
    
    print(f"Registros con 'exceso' en Evento: {exceso_mask.sum()}")
    print(f"Textos únicos con 'exceso': {len(textos_con_exceso)}")
    
    # Mostrar primeros 5 textos
    for i, texto in enumerate(textos_con_exceso[:5]):
        print(f"\n  Texto {i+1}: '{texto}'")
        # Probar regex
        match = SPEED_EXCESS_RE.search(texto)
        if match:
            print(f"    ✓ REGEX MATCH: velocidad={match.group(1)}, zona={match.group(2)}")
        else:
            print(f"    ✗ NO MATCH con regex")
    
    # Aplicar detección completa
    df["is_exceso"] = df["Evento_clean"].apply(lambda t: bool(SPEED_EXCESS_RE.search(t)))
    
    # Análisis de corridas
    df["run_id"] = (df["is_exceso"] != df["is_exceso"].shift(1)).cumsum()
    
    corridas_exceso = []
    duracion_total = 0
    
    for run_id, group in df.groupby("run_id"):
        if group["is_exceso"].iloc[0]:
            corridas_exceso.append(len(group))
            
            # Calcular duración
            start_ts = group["Hora"].iloc[0]
            end_idx = group.index.max()
            
            if end_idx + 1 in df.index:
                end_ts = df.at[end_idx + 1, "Hora"]
                raw_sec = (end_ts - start_ts).total_seconds()
            else:
                raw_sec = group["dt_sec"].sum()
                
            # Buscar "durante X minutos"
            mins_match = group["Evento_clean"].str.extract(MINUTES_RE)
            if mins_match[0].notna().any():
                mins = int(mins_match[0].dropna().astype(int).max())
                raw_sec = mins * 60 + raw_sec
                
            duracion_total += raw_sec
    
    print(f"\n--- RESUMEN DE DETECCIÓN ---")
    print(f"Registros marcados como exceso: {df['is_exceso'].sum()}")
    print(f"Número de corridas de exceso: {len(corridas_exceso)}")
    print(f"Duración total estimada: {duracion_total:.0f} segundos")
    if corridas_exceso:
        print(f"Tamaños de corridas: {corridas_exceso[:10]}...")  # Primeras 10
    
    # Verificar columnas de Evento y Flags por separado
    print(f"\n--- ANÁLISIS SEPARADO DE COLUMNAS ---")
    evento_excesos = df["Evento"].fillna("").str.contains("exceso", case=False, na=False).sum()
    flags_excesos = df["Flags"].fillna("").str.contains("exceso", case=False, na=False).sum()
    print(f"Excesos en columna Evento: {evento_excesos}")
    print(f"Excesos en columna Flags: {flags_excesos}")
    
    # Convertir columna Hora para análisis de duración
    df["Hora"] = pd.to_datetime(df["Hora"], format="%d/%m/%Y %H:%M:%S")
    df["dt_sec"] = df["Hora"].diff().dt.total_seconds().fillna(0)
    
    return df

# Ejecutar debug
if __name__ == "__main__":
    excel_files = list(DATA_DIR.glob("*.xlsx"))
    print(f"Archivos Excel encontrados: {len(excel_files)}")
    
    # Debuggear el primer archivo que encuentre
    if excel_files:
        # Buscar específicamente el archivo del 5 de junio si existe
        target_file = None
        for f in excel_files:
            if "2025-06-05" in f.name or "05-06-2025" in f.name or "20250605" in f.name:
                target_file = f
                break
        
        if not target_file:
            target_file = excel_files[0]
            
        debug_single_file(target_file)
    else:
        print("No se encontraron archivos Excel en el directorio")