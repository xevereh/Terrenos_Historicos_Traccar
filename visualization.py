"""
visualization.py - Funciones de visualización para Streamlit
Mejorado con mejor manejo de GPS y visualizaciones
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import config

def create_cluster_scatter(df_metrics, x_col="vel_media_mov", y_col="num_excesos"):
    """Crea scatter plot de clusters"""
    fig = px.scatter(
        df_metrics,
        x=x_col,
        y=y_col,
        color="perfil",
        color_discrete_map={
            config.CLUSTER_NAMES[i]: config.CLUSTER_COLORS[i] 
            for i in range(config.N_CLUSTERS)
        },
        size="distancia_km",
        hover_data=["fecha", "distancia_km", "excesos_por_km"],
        title="Análisis de Clusters de Conducción",
        labels={
            "vel_media_mov": "Velocidad Media (km/h)",
            "num_excesos": "Número de Excesos",
            "distancia_km": "Distancia (km)"
        }
    )
    
    fig.update_layout(
        height=500,
        hovermode='closest',
        showlegend=True
    )
    
    return fig

def create_metrics_radar(row_metrics, cluster_centers=None):
    """Crea gráfico radar de métricas del día"""
    categories = [
        'Vel. Media', 'Vel. Máx', 'Excesos/km', 
        'Acel. Bruscas', 'Fren. Bruscas'
    ]
    
    # Normalizar valores (0-100)
    values = [
        min(row_metrics["vel_media_mov"] / 120 * 100, 100),
        min(row_metrics["vel_max"] / 150 * 100, 100),
        min(row_metrics["excesos_por_km"] * 50, 100),
        min(row_metrics["harsh_accel_windows"] * 10, 100),
        min(row_metrics["harsh_brake_windows"] * 10, 100)
    ]
    
    fig = go.Figure()
    
    # Añadir datos del día
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=f'Día {row_metrics["fecha"]}',
        line_color=config.CLUSTER_COLORS[row_metrics.get("cluster", 0)]
    ))
    
    # Si hay centroides, añadir el promedio del cluster
    if cluster_centers is not None and "cluster" in row_metrics:
        cluster_avg = cluster_centers.iloc[row_metrics["cluster"]]
        cluster_values = [
            min(cluster_avg["vel_media_mov"] / 120 * 100, 100),
            min(cluster_avg["vel_max"] / 150 * 100, 100),
            min(cluster_avg["excesos_por_km"] * 50, 100),
            min(cluster_avg["harsh_accel_windows"] * 10, 100),
            min(cluster_avg["harsh_brake_windows"] * 10, 100)
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=cluster_values,
            theta=categories,
            fill='toself',
            name=f'Promedio {row_metrics["perfil"]}',
            line_color='gray',
            opacity=0.3
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Perfil de Conducción - Análisis Radar",
        height=400
    )
    
    return fig

def create_daily_timeline(df_day):
    """Crea línea de tiempo de velocidad del día"""
    fig = go.Figure()
    
    # Línea de velocidad
    fig.add_trace(go.Scatter(
        x=df_day["Hora"],
        y=df_day["Vel_kmh"],
        mode='lines',
        name='Velocidad',
        line=dict(color='blue', width=2)
    ))
    
    # Marcar excesos si existen
    if "is_exceso" in df_day.columns:
        excesos = df_day[df_day["is_exceso"]]
        if not excesos.empty:
            fig.add_trace(go.Scatter(
                x=excesos["Hora"],
                y=excesos["Vel_kmh"],
                mode='markers',
                name='Excesos',
                marker=dict(color='red', size=8, symbol='x')
            ))
    
    # Añadir líneas de referencia
    fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                  annotation_text="80 km/h")
    fig.add_hline(y=120, line_dash="dash", line_color="red", 
                  annotation_text="120 km/h")
    
    fig.update_layout(
        title="Velocidad a lo largo del día",
        xaxis_title="Hora",
        yaxis_title="Velocidad (km/h)",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_hourly_heatmap(files_data):
    """Crea heatmap de excesos por hora del día"""
    # Preparar matriz hora x día
    hours = list(range(24))
    dates = sorted(files_data.keys())
    
    matrix = np.zeros((24, len(dates)))
    
    for i, date in enumerate(dates):
        df = files_data[date]["df"]
        if "is_exceso" in df.columns:
            excesos = df[df["is_exceso"]]
            
            for hour in hours:
                count = len(excesos[excesos["Hora"].dt.hour == hour])
                matrix[hour, i] = count
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[str(d) for d in dates],
        y=[f"{h:02d}:00" for h in hours],
        colorscale="Reds",
        text=matrix,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Distribución de Excesos por Hora",
        xaxis_title="Fecha",
        yaxis_title="Hora del día",
        height=500
    )
    
    return fig

def create_metrics_bars(row_metrics, df_all):
    """Crea gráfico de barras comparando métricas del día vs promedio"""
    metrics = {
        "Velocidad Media": {
            "dia": row_metrics["vel_media_mov"],
            "promedio": df_all["vel_media_mov"].mean(),
            "unidad": "km/h"
        },
        "Excesos": {
            "dia": row_metrics["num_excesos"],
            "promedio": df_all["num_excesos"].mean(),
            "unidad": ""
        },
        "Ratio Excesos/km": {
            "dia": row_metrics["excesos_por_km"],
            "promedio": df_all["excesos_por_km"].mean(),
            "unidad": ""
        },
        "Acel. Bruscas": {
            "dia": row_metrics["harsh_accel_windows"],
            "promedio": df_all["harsh_accel_windows"].mean(),
            "unidad": "ventanas"
        },
        "Fren. Bruscas": {
            "dia": row_metrics["harsh_brake_windows"],
            "promedio": df_all["harsh_brake_windows"].mean(),
            "unidad": "ventanas"
        }
    }
    
    fig = go.Figure()
    
    # Barras del día
    fig.add_trace(go.Bar(
        x=list(metrics.keys()),
        y=[m["dia"] for m in metrics.values()],
        name=f'Día {row_metrics["fecha"]}',
        marker_color=config.CLUSTER_COLORS[row_metrics.get("cluster", 0)]
    ))
    
    # Línea del promedio
    fig.add_trace(go.Scatter(
        x=list(metrics.keys()),
        y=[m["promedio"] for m in metrics.values()],
        mode='lines+markers',
        name='Promedio General',
        line=dict(color='black', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Comparación de Métricas vs Promedio",
        yaxis_title="Valor",
        height=400,
        showlegend=True
    )
    
    return fig

def create_map_view(df_day, max_speed_kmh=200):
    """
    Crea mapa del recorrido si hay coordenadas GPS
    Usa estrategia mejorada para filtrar coordenadas erróneas
    """
    if "Latitud" not in df_day.columns or "Longitud" not in df_day.columns:
        return None, None
    
    # Filtrar coordenadas válidas
    df_map = df_day.dropna(subset=["Latitud", "Longitud"]).copy()
    
    if df_map.empty:
        return None, None
    
    # Función para calcular distancia entre puntos
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Radio de la Tierra en km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    # Estrategia mejorada: usar velocidad para validar saltos
    valid_indices = []
    total_distance = 0
    problematic_jumps = []
    
    for i in range(len(df_map)):
        if i == 0:
            valid_indices.append(i)
            continue
        
        # Buscar el último punto válido
        if not valid_indices:
            continue
            
        last_valid_idx = valid_indices[-1]
        
        # Calcular distancia y tiempo desde el último punto válido
        dist = haversine(
            df_map.iloc[last_valid_idx]["Latitud"],
            df_map.iloc[last_valid_idx]["Longitud"],
            df_map.iloc[i]["Latitud"],
            df_map.iloc[i]["Longitud"]
        )
        
        time_diff = (df_map.iloc[i]["Hora"] - df_map.iloc[last_valid_idx]["Hora"]).total_seconds()
        
        # Calcular velocidad implícita del salto
        if time_diff > 0:
            implied_speed = (dist / time_diff) * 3600  # km/h
        else:
            implied_speed = float('inf')
        
        # Validar el punto:
        # 1. Si la velocidad implícita es razonable (menor a max_speed_kmh)
        # 2. O si la distancia es muy pequeña (menos de 0.1 km)
        # 3. O si han pasado más de 5 minutos (podría ser una pausa)
        if implied_speed <= max_speed_kmh or dist < 0.1 or time_diff > 300:
            valid_indices.append(i)
            total_distance += dist
        else:
            problematic_jumps.append({
                'from_idx': last_valid_idx,
                'to_idx': i,
                'distance': dist,
                'implied_speed': implied_speed
            })
    
    # Filtrar DataFrame con índices válidos
    df_map_clean = df_map.iloc[valid_indices].reset_index(drop=True)
    
    print(f"Puntos GPS totales: {len(df_map)}")
    print(f"Puntos válidos: {len(df_map_clean)}")
    print(f"Saltos problemáticos eliminados: {len(problematic_jumps)}")
    
    # Crear mapa con colores tipo semáforo
    fig = px.scatter_mapbox(
        df_map_clean,
        lat="Latitud",
        lon="Longitud",
        color="Vel_kmh",
        size="Vel_kmh",
        color_continuous_scale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
        size_max=15,
        zoom=10,
        height=600,
        title=f"Recorrido GPS del día (Distancia real: {total_distance:.1f} km)"
    )
    
    # Añadir línea de trayectoria con segmentos
    # Dividir en segmentos cuando hay gaps grandes de tiempo
    segments = []
    current_segment = [0]
    
    for i in range(1, len(df_map_clean)):
        time_gap = (df_map_clean.iloc[i]["Hora"] - df_map_clean.iloc[i-1]["Hora"]).total_seconds()
        if time_gap > 300:  # Más de 5 minutos
            segments.append(current_segment)
            current_segment = [i]
        else:
            current_segment.append(i)
    segments.append(current_segment)
    
    # Dibujar cada segmento
    for seg_idx, segment in enumerate(segments):
        if len(segment) > 1:
            seg_df = df_map_clean.iloc[segment]
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=seg_df["Longitud"],
                lat=seg_df["Latitud"],
                line=dict(width=3, color="blue"),
                name=f"Trayectoria {seg_idx+1}" if len(segments) > 1 else "Trayectoria",
                showlegend=True
            ))
    
    # Marcar puntos de exceso
    if "is_exceso" in df_map_clean.columns:
        excesos = df_map_clean[df_map_clean["is_exceso"]]
        if not excesos.empty:
            fig.add_trace(go.Scattermapbox(
                mode="markers",
                lon=excesos["Longitud"],
                lat=excesos["Latitud"],
                marker=dict(size=10, color="red", symbol="x"),
                name="Excesos de velocidad",
                showlegend=True
            ))
    
    # Actualizar layout con mejor estilo
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(
                lat=df_map_clean["Latitud"].mean(),
                lon=df_map_clean["Longitud"].mean()
            ),
            zoom=12
        ),
        coloraxis_colorbar=dict(
            title="Velocidad<br>(km/h)",
            tickvals=[0, 40, 80, 120],
            ticktext=["0", "40", "80", "120+"]
        )
    )
    
    return fig, total_distance