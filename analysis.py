"""
analysis.py - Clustering y generación de narrativas con LLM MODIFICADO
Cambiado de K-Means relativo a umbrales absolutos
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from openai import OpenAI
import config

def perform_clustering(df_metrics):
    """
    CORREGIDO: Aplica clustering inteligente basado en distribución de risk_score
    del rango seleccionado, sin forzar porcentajes fijos
    """
    print(f"Aplicando clustering inteligente basado en distribución...")
    
    if df_metrics.empty:
        return df_metrics, pd.DataFrame()
    
    # Calcular percentiles del risk_score para este conjunto de datos
    scores = df_metrics["risk_score"].values
    
    # Si todos los valores son muy similares (diferencia < 10), todo es "Seguro"
    score_range = scores.max() - scores.min()
    if score_range < 10:
        df_metrics["cluster"] = 0  # Todo seguro
        df_metrics["perfil"] = "Conducción Segura"
        print(f"Todos los días clasificados como 'Seguro' (rango muy pequeño: {score_range:.1f})")
    else:
        # Usar percentiles del conjunto para crear umbrales dinámicos
        p33 = np.percentile(scores, 33)
        p67 = np.percentile(scores, 67)
        
        # Pero ajustar si los percentiles son demasiado bajos
        min_moderate_threshold = 15  # Mínimo para ser "moderado"
        min_risky_threshold = 30     # Mínimo para ser "arriesgado"
        
        # Ajustar umbrales
        threshold_low = max(p33, min_moderate_threshold)
        threshold_high = max(p67, min_risky_threshold)
        
        # Asegurar que threshold_high > threshold_low
        if threshold_high <= threshold_low:
            threshold_high = threshold_low + 10
        
        def assign_smart_cluster(risk_score):
            if risk_score < threshold_low:
                return 0  # Seguro
            elif risk_score < threshold_high:
                return 1  # Moderado
            else:
                return 2  # Arriesgado
        
        df_metrics["cluster"] = df_metrics["risk_score"].apply(assign_smart_cluster)
        df_metrics["perfil"] = df_metrics["cluster"].map(config.CLUSTER_NAMES)
        
        print(f"Umbrales calculados: Seguro < {threshold_low:.1f}, Moderado < {threshold_high:.1f}, Arriesgado >= {threshold_high:.1f}")
    
    # Calcular centroides para visualización
    features = [
        "vel_media_mov", "vel_max", "distancia_km",
        "num_excesos", "dur_exceso_tot_sec", "excesos_por_km",
        "harsh_accel_windows", "harsh_brake_windows"
    ]
    
    cluster_centers = []
    for cluster_id in range(config.N_CLUSTERS):
        cluster_data = df_metrics[df_metrics["cluster"] == cluster_id]
        if len(cluster_data) > 0:
            center = cluster_data[features].mean().values
        else:
            center = np.zeros(len(features))
        cluster_centers.append(center)
    
    cluster_centers = pd.DataFrame(cluster_centers, columns=features)
    
    # Mostrar estadísticas del clustering
    cluster_counts = df_metrics["perfil"].value_counts()
    print(f"Distribución de clusters:")
    for perfil, count in cluster_counts.items():
        percentage = (count / len(df_metrics)) * 100
        print(f"  {perfil}: {count} días ({percentage:.1f}%)")
    
    return df_metrics, cluster_centers

def calculate_enhanced_risk_score(row, speed_threshold=None):
    """
    CORREGIDO: Cálculo mejorado de risk_score que SÍ usa el umbral configurable
    """
    if speed_threshold is None:
        speed_threshold = config.DEFAULT_SPEED_THRESHOLD
    
    # Pesos para cada factor
    weights = {
        "excesos_oficiales": 0.25,      # 25% - Excesos detectados por texto
        "velocidad_maxima": 0.15,       # 15% - Vel máxima sobre 80 km/h
        "agresividad": 0.25,            # 25% - Acel + frenadas bruscas
        "duracion_excesos": 0.15,       # 15% - Tiempo en excesos oficiales
        "velocidad_sobre_umbral": 0.20  # 20% - NUEVO: Factor de umbral empresa
    }
    
    # Factores existentes
    excesos_score = min(row["excesos_por_km"] * 100, 100)
    vel_score = min((row["vel_max"] - 80) / 40 * 100, 100) if row["vel_max"] > 80 else 0
    agresiv_score = min((row["harsh_accel_windows"] + row["harsh_brake_windows"]) * 5, 100)
    duracion_score = min(row["dur_exceso_tot_sec"] / 600 * 100, 100)
    
    # NUEVO: Factor de velocidad sobre umbral empresa
    if row["vel_max"] > speed_threshold:
        # Penalización progresiva por exceder umbral empresa
        excess_speed = row["vel_max"] - speed_threshold
        umbral_score = min((excess_speed / 30) * 100, 100)  # 30 km/h de exceso = 100% penalización
    else:
        umbral_score = 0
    
    # Calcular score final
    risk_score = (
        excesos_score * weights["excesos_oficiales"] +
        vel_score * weights["velocidad_maxima"] +
        agresiv_score * weights["agresividad"] +
        duracion_score * weights["duracion_excesos"] +
        umbral_score * weights["velocidad_sobre_umbral"]
    )
    
    return min(risk_score, 100)

def calculate_risk_score(row):
    """
    MANTENER función original para compatibilidad
    """
    # Ponderaciones para cada factor
    weights = {
        "excesos": 0.3,
        "velocidad": 0.2,
        "agresividad": 0.3,
        "duracion": 0.2
    }
    
    # Normalizar métricas (0-100)
    excesos_score = min(row["excesos_por_km"] * 100, 100)
    vel_score = min((row["vel_max"] - 80) / 40 * 100, 100) if row["vel_max"] > 80 else 0
    agresiv_score = min((row["harsh_accel_windows"] + row["harsh_brake_windows"]) * 5, 100)
    duracion_score = min(row["dur_exceso_tot_sec"] / 600 * 100, 100)  # 10 min = 100
    
    risk_score = (
        excesos_score * weights["excesos"] +
        vel_score * weights["velocidad"] +
        agresiv_score * weights["agresividad"] +
        duracion_score * weights["duracion"]
    )
    
    return min(risk_score, 100)

def generate_narrative(row, client=None):
    """Genera narrativa descriptiva usando LLM (SIN CAMBIOS)"""
    if client is None:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    # Construir prompt con las métricas
    prompt = f"""Analiza el siguiente perfil de conducción del día {row['fecha']}:

Distancia recorrida: {row['distancia_km']:.1f} km
Velocidad media en movimiento: {row['vel_media_mov']:.1f} km/h
Velocidad máxima: {row['vel_max']:.1f} km/h
Número de excesos de velocidad: {row['num_excesos']}
Duración total de excesos: {row['dur_exceso_tot_sec']:.0f} segundos
Ratio de excesos por km: {row['excesos_por_km']:.2f}
Ventanas con aceleraciones bruscas: {row['harsh_accel_windows']}
Ventanas con frenadas bruscas: {row['harsh_brake_windows']}
Tiempo total de conducción: {row.get('tiempo_conduccion_min', 0):.1f} minutos

Genera un párrafo breve (3-4 líneas) describiendo este perfil de conducción.
Enfócate en los aspectos más relevantes y da una evaluación general del estilo de manejo.
No uses viñetas ni listas, solo texto corrido."""

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generando narrativa: {str(e)}"

def analyze_cluster_characteristics(df_metrics, cluster_centers):
    """Analiza las características de cada cluster (SIN CAMBIOS)"""
    cluster_analysis = {}
    
    for cluster_id in range(config.N_CLUSTERS):
        cluster_data = df_metrics[df_metrics["cluster"] == cluster_id]
        
        if len(cluster_data) == 0:
            continue
            
        analysis = {
            "nombre": config.CLUSTER_NAMES[cluster_id],
            "num_dias": len(cluster_data),
            "caracteristicas": {
                "vel_media_promedio": cluster_data["vel_media_mov"].mean(),
                "distancia_promedio": cluster_data["distancia_km"].mean(),
                "excesos_promedio": cluster_data["num_excesos"].mean(),
                "ratio_excesos_promedio": cluster_data["excesos_por_km"].mean(),
                "agresividad_accel": cluster_data["harsh_accel_windows"].mean(),
                "agresividad_freno": cluster_data["harsh_brake_windows"].mean()
            },
            "dias": cluster_data["fecha"].tolist()
        }
        
        cluster_analysis[cluster_id] = analysis
    
    return cluster_analysis

def find_max_speed_day(df_metrics):
    """
    NUEVA función: Encuentra el día con la velocidad máxima registrada
    """
    if df_metrics.empty:
        return None, None
    
    max_speed_row = df_metrics.loc[df_metrics['vel_max'].idxmax()]
    return max_speed_row['fecha'], max_speed_row['vel_max']