"""
main.py - Aplicación Streamlit MODIFICADA para análisis de conducción
Adaptada para múltiples vehículos y rangos de fechas
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta
from openai import OpenAI

# Importar módulos locales
import config
from data_processing import (
    discover_vehicles, 
    process_vehicle_files,
    process_all_files  # Mantener para compatibilidad
)
from analysis import (
    perform_clustering, 
    generate_narrative, 
    analyze_cluster_characteristics,
    calculate_risk_score,
    calculate_enhanced_risk_score,  # NUEVA función
    find_max_speed_day
)
from visualization import (
    create_cluster_scatter,
    create_metrics_radar,
    create_daily_timeline,
    create_hourly_heatmap,
    create_metrics_bars,
    create_map_view
)

# Configuración de página
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

# CSS personalizado (SIN CAMBIOS)
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .narrative-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .vehicle-info {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .date-range-info {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🚗 DriveTech - Sistema de Análisis de Conducción Multi-Vehículo")
st.markdown("---")

# Inicializar estado de sesión - AMPLIADO
if 'vehicles_discovered' not in st.session_state:
    st.session_state.vehicles_discovered = False
    st.session_state.available_vehicles = {}
    st.session_state.selected_vehicle = None
    st.session_state.vehicle_data_loaded = False
    st.session_state.df_metrics = None
    st.session_state.files_data = None
    st.session_state.cluster_centers = None
    st.session_state.openai_client = None
    st.session_state.selected_start_date = None
    st.session_state.selected_end_date = None
    st.session_state.available_dates_in_range = []

# Función para calcular resumen de intervalo
def calculate_interval_summary(df_metrics):
    """Calcula resumen agregado para un intervalo de fechas"""
    if df_metrics.empty:
        return None
    
    # Agregar métricas del intervalo
    summary = {
        'fecha_inicio': df_metrics['fecha'].min(),
        'fecha_fin': df_metrics['fecha'].max(),
        'num_dias': len(df_metrics),
        'distancia_total': df_metrics['distancia_km'].sum(),
        'vel_media_promedio': df_metrics['vel_media_mov'].mean(),
        'vel_max_absoluta': df_metrics['vel_max'].max(),
        'excesos_totales': df_metrics['num_excesos'].sum(),
        'dur_exceso_total': df_metrics['dur_exceso_tot_sec'].sum(),
        'excesos_por_km_promedio': df_metrics['excesos_por_km'].mean(),
        'harsh_accel_total': df_metrics['harsh_accel_windows'].sum(),
        'harsh_brake_total': df_metrics['harsh_brake_windows'].sum(),
        'risk_score_promedio': df_metrics['risk_score'].mean(),
        'risk_score_maximo': df_metrics['risk_score'].max(),
        'cluster_predominante': df_metrics['perfil'].mode().iloc[0] if not df_metrics['perfil'].mode().empty else "N/A"
    }
    
    return summary

# Sidebar - MODIFICADO COMPLETAMENTE
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # API Key
    api_key = st.text_input(
        "OpenAI API Key",
        value=config.OPENAI_API_KEY if config.OPENAI_API_KEY != "tu-api-key-aqui" else "",
        type="password",
        help="Ingresa tu API key de OpenAI para generar narrativas"
    )
    
    if api_key:
        st.session_state.openai_client = OpenAI(api_key=api_key)
    
    st.markdown("---")
    
    # PASO 1: Descubrir vehículos
    st.header("🔍 Paso 1: Descubrir Vehículos")
    
    if st.button("🔄 Escanear Vehículos Disponibles", type="primary"):
        with st.spinner("Escaneando estructura de carpetas..."):
            try:
                vehicles = discover_vehicles()
                if vehicles:
                    st.session_state.available_vehicles = vehicles
                    st.session_state.vehicles_discovered = True
                    st.success(f"✅ {len(vehicles)} vehículos encontrados")
                    
                    # Mostrar resumen de vehículos
                    for vehicle_id, info in vehicles.items():
                        st.info(f"🚚 **{vehicle_id}**: {info['total_days']} días disponibles")
                else:
                    st.error("No se encontraron vehículos en la estructura de carpetas")
            except Exception as e:
                st.error(f"Error al escanear vehículos: {str(e)}")
    
    # PASO 2: Selección de vehículo (solo si hay vehículos descubiertos)
    if st.session_state.vehicles_discovered:
        st.markdown("---")
        st.header("🚚 Paso 2: Seleccionar Vehículo")
        
        vehicle_options = list(st.session_state.available_vehicles.keys())
        selected_vehicle = st.selectbox(
            "Patente del vehículo:",
            options=vehicle_options,
            help="Selecciona el vehículo a analizar"
        )
        
        if selected_vehicle:
            st.session_state.selected_vehicle = selected_vehicle
            vehicle_info = st.session_state.available_vehicles[selected_vehicle]
            
            # Mostrar info del vehículo seleccionado
            st.markdown(
                f'<div class="vehicle-info">'
                f'<strong>Vehículo seleccionado:</strong> {selected_vehicle}<br>'
                f'<strong>Días disponibles:</strong> {vehicle_info["total_days"]}<br>'
                f'<strong>Rango de datos:</strong> {vehicle_info["date_range"][0]} a {vehicle_info["date_range"][1]}'
                f'</div>',
                unsafe_allow_html=True
            )
        
        # PASO 3: Selección de rango de fechas del terreno
        st.markdown("---")
        st.header("📅 Paso 3: Rango de Fechas del Terreno")
        
        if st.session_state.selected_vehicle:
            vehicle_info = st.session_state.available_vehicles[st.session_state.selected_vehicle]
            min_date = vehicle_info["date_range"][0]
            max_date = vehicle_info["date_range"][1]
            
            # Selectores de fecha - CORREGIDO para evitar conflictos
            col1, col2 = st.columns(2)
            
            with col1:
                # Usar min_date como valor por defecto siempre
                start_date = st.date_input(
                    "Fecha de salida (inicio):",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    help="Fecha de inicio del terreno",
                    key=f"start_date_{st.session_state.selected_vehicle}"
                )
            
            with col2:
                # Calcular valor por defecto válido para end_date
                default_end = min(start_date + timedelta(days=7), max_date)
                end_date = st.date_input(
                    "Fecha de retorno (fin):",
                    value=default_end,
                    min_value=start_date,
                    max_value=max_date,
                    help="Fecha de fin del terreno",
                    key=f"end_date_{st.session_state.selected_vehicle}"
                )
            
            # Validar rango
            if start_date <= end_date:
                st.session_state.selected_start_date = start_date
                st.session_state.selected_end_date = end_date
                
                days_selected = (end_date - start_date).days + 1
                st.markdown(
                    f'<div class="date-range-info">'
                    f'<strong>Rango seleccionado:</strong> {days_selected} días<br>'
                    f'<strong>Desde:</strong> {start_date.strftime("%d/%m/%Y")}<br>'
                    f'<strong>Hasta:</strong> {end_date.strftime("%d/%m/%Y")}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.error("❌ La fecha de fin debe ser posterior a la fecha de inicio")
        
        # PASO 4: Procesar datos del vehículo
        st.markdown("---")
        st.header("⚡ Paso 4: Procesar Datos")
        
        if (st.session_state.selected_vehicle and 
            st.session_state.selected_start_date and 
            st.session_state.selected_end_date):
            
            if st.button("🔄 Procesar Datos del Terreno", type="secondary"):
                with st.spinner("Procesando archivos del vehículo..."):
                    try:
                        # Procesar archivos del vehículo en el rango
                        df_metrics, files_data = process_vehicle_files(
                            st.session_state.selected_vehicle,
                            st.session_state.selected_start_date,
                            st.session_state.selected_end_date
                        )
                        
                        if len(df_metrics) == 0:
                            st.error("No se encontraron datos para el vehículo y rango seleccionado")
                        else:
                            # Calcular risk score
                            df_metrics["risk_score"] = df_metrics.apply(calculate_risk_score, axis=1)
                            
                            # Aplicar clustering
                            df_metrics, cluster_centers = perform_clustering(df_metrics)
                            
                            # Guardar en session state
                            st.session_state.df_metrics = df_metrics
                            st.session_state.files_data = files_data
                            st.session_state.cluster_centers = cluster_centers
                            st.session_state.vehicle_data_loaded = True
                            
                            # Obtener fechas disponibles para selector individual
                            st.session_state.available_dates_in_range = sorted(df_metrics["fecha"].unique())
                            
                            st.success(f"✅ {len(df_metrics)} días procesados para {st.session_state.selected_vehicle}")
                            
                            # Mostrar resumen rápido
                            with st.expander("📊 Vista previa de datos procesados"):
                                st.dataframe(
                                    df_metrics[[
                                        "fecha", "num_excesos", "vel_media_mov", 
                                        "vel_max", "distancia_km", "risk_score", "perfil"
                                    ]].head()
                                )
                    
                    except Exception as e:
                        st.error(f"Error al procesar datos: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # PASO 5: Filtros adicionales (solo si hay datos cargados)
    if st.session_state.vehicle_data_loaded:
        st.markdown("---")
        st.header("🔍 Paso 5: Análisis Detallado")
        
        # NUEVO: Configuración de umbral de velocidad
        st.markdown("### ⚙️ Configuración de Análisis")
        
        speed_threshold = st.slider(
            "Umbral de velocidad de empresa (km/h):",
            min_value=60,
            max_value=120,
            value=config.DEFAULT_SPEED_THRESHOLD,
            step=5,
            key="speed_threshold_slider"
        )
        
        # CORREGIDO: Recalcular automáticamente cuando cambia el umbral
        if speed_threshold != config.DEFAULT_SPEED_THRESHOLD:
            st.warning(f"⚠️ **Umbral modificado:** {speed_threshold} km/h (por defecto: {config.DEFAULT_SPEED_THRESHOLD} km/h)")
            
            # Recalcular risk_score con nuevo umbral
            if st.session_state.df_metrics is not None:
                with st.spinner("Recalculando con nuevo umbral..."):
                    # Recalcular risk scores
                    st.session_state.df_metrics["risk_score"] = st.session_state.df_metrics.apply(
                        lambda row: calculate_enhanced_risk_score(row, speed_threshold), axis=1
                    )
                    
                    # Recalcular clustering
                    st.session_state.df_metrics, st.session_state.cluster_centers = perform_clustering(
                        st.session_state.df_metrics
                    )
                    
                    st.success(f"✅ Datos recalculados con umbral de {speed_threshold} km/h")
        
        st.markdown("---")
        
        # Selector de día específico dentro del rango
        if st.session_state.available_dates_in_range:
            selected_date = st.selectbox(
                "Seleccionar día específico:",
                options=st.session_state.available_dates_in_range,
                format_func=lambda x: x.strftime("%d/%m/%Y"),
                help="Selecciona un día específico para análisis detallado"
            )
        
        # Filtro por perfil
        if st.session_state.df_metrics is not None:
            perfiles = ["Todos"] + list(st.session_state.df_metrics["perfil"].unique())
            perfil_filter = st.selectbox("Filtrar por perfil", perfiles)

# Contenido principal - MODIFICADO
if not st.session_state.vehicles_discovered:
    st.info("👈 Comienza escaneando los vehículos disponibles en la barra lateral")
    st.markdown("""
    ### Nuevo Sistema Multi-Vehículo 🚚
    
    **Pasos para usar el sistema:**
    1. **Escanear vehículos** - Detecta automáticamente las patentes disponibles
    2. **Seleccionar vehículo** - Elige la patente a analizar  
    3. **Definir rango de fechas** - Establece el período del terreno
    4. **Procesar datos** - Carga y analiza la información
    5. **Análisis detallado** - Explora resúmenes y días específicos
    
    ### Estructura de Datos Esperada 📁
    ```
    Resumen/
    ├── TWJL30_2025-04-01_2025-06-23/
    │   ├── 2025-04-01_TWJL30/
    │   │   └── 2025-04-01_TWJL30.xlsx
    │   └── 2025-04-02_TWJL30/
    └── OTRA_PATENTE_2025-04-01_2025-06-23/
    ```
    """)

elif not st.session_state.vehicle_data_loaded:
    st.info("👈 Selecciona un vehículo y procesa los datos para continuar")
    
    # Mostrar vehículos disponibles
    if st.session_state.available_vehicles:
        st.markdown("### 🚚 Vehículos Disponibles")
        
        cols = st.columns(min(3, len(st.session_state.available_vehicles)))
        for i, (vehicle_id, info) in enumerate(st.session_state.available_vehicles.items()):
            with cols[i % 3]:
                st.markdown(
                    f"<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 5px;'>"
                    f"<h4>🚚 {vehicle_id}</h4>"
                    f"<p><strong>{info['total_days']}</strong> días disponibles</p>"
                    f"<p><small>{info['date_range'][0]} → {info['date_range'][1]}</small></p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

else:
    # Datos cargados - mostrar análisis
    
    # Calcular resumen del intervalo
    interval_summary = calculate_interval_summary(st.session_state.df_metrics)
    
    # Aplicar filtro de perfil si es necesario
    if 'perfil_filter' in locals() and perfil_filter != "Todos":
        df_filtered = st.session_state.df_metrics[
            st.session_state.df_metrics["perfil"] == perfil_filter
        ]
    else:
        df_filtered = st.session_state.df_metrics
    
    # Tab layout - MODIFICADO
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumen del Terreno", 
        "📋 Análisis por Día",
        "📈 Análisis de Clusters", 
        "🗺️ Visualizaciones",
        "📋 Comparativas"
    ])
    
    with tab1:
        # NUEVA TAB: Resumen del intervalo completo
        st.header(f"Resumen del Terreno - Vehículo {st.session_state.selected_vehicle}")
        
        if interval_summary:
            # Información del terreno
            st.markdown(
                f'<div class="vehicle-info">'
                f'<strong>📅 Período:</strong> {interval_summary["fecha_inicio"].strftime("%d/%m/%Y")} → {interval_summary["fecha_fin"].strftime("%d/%m/%Y")}<br>'
                f'<strong>🚚 Vehículo:</strong> {st.session_state.selected_vehicle}<br>'
                f'<strong>📊 Días analizados:</strong> {interval_summary["num_dias"]}'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Calcular información adicional del terreno
            max_speed_date, max_speed_value = find_max_speed_day(df_filtered)
            
            # Métricas principales del terreno
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Distancia Total del Terreno",
                    f"{interval_summary['distancia_total']:.1f} km",
                    help="Suma de todas las distancias recorridas durante el terreno"
                )
                
            with col2:
                st.metric(
                    "Velocidad Media del Terreno",
                    f"{interval_summary['vel_media_promedio']:.1f} km/h",
                    help="Promedio de velocidades medias de todos los días"
                )
                
            with col3:
                st.metric(
                    "Excesos Totales",
                    f"{interval_summary['excesos_totales']}",
                    delta_color="inverse",
                    help="Suma de todos los excesos de velocidad del terreno"
                )
                
            with col4:
                # Determinar color del score promedio
                avg_risk = interval_summary['risk_score_promedio']
                if avg_risk < 30:
                    risk_color = "🟢"
                elif avg_risk < 60:
                    risk_color = "🟡"
                else:
                    risk_color = "🔴"
                    
                st.metric(
                    "Score Promedio del Terreno",
                    f"{risk_color} {avg_risk:.0f}/100",
                    delta=f"Máx: {interval_summary['risk_score_maximo']:.0f}",
                    help="Score de riesgo promedio y máximo del terreno"
                )
            
            # Segunda fila de métricas CON INDICADOR DE DÍA
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                # NUEVO: Mostrar día de velocidad máxima
                if max_speed_date:
                    st.metric(
                        "Velocidad Máxima Registrada",
                        f"{interval_summary['vel_max_absoluta']:.0f} km/h",
                        help=f"📅 Registrada el {max_speed_date.strftime('%d/%m/%Y')}"
                    )
                    # Mensaje adicional más visible
                    st.caption(f"📅 **Día:** {max_speed_date.strftime('%d/%m/%Y')}")
                else:
                    st.metric(
                        "Velocidad Máxima Registrada",
                        f"{interval_summary['vel_max_absoluta']:.0f} km/h"
                    )
                
            with col6:
                st.metric(
                    "Tiempo Total en Excesos",
                    f"{interval_summary['dur_exceso_total']:.0f} seg"
                )
                
            with col7:
                st.metric(
                    "Aceleraciones Bruscas Total",
                    f"{interval_summary['harsh_accel_total']}",
                    delta_color="inverse"
                )
                
            with col8:
                st.metric(
                    "Frenadas Bruscas Total",
                    f"{interval_summary['harsh_brake_total']}",
                    delta_color="inverse"
                )
            
            st.markdown("---")
            
            # Perfil predominante del terreno
            st.markdown(f"### 🎯 Perfil Predominante del Terreno: **{interval_summary['cluster_predominante']}**")
            
            # Gráficos del terreno
            col_graf1, col_graf2 = st.columns(2)
            
            with col_graf1:
                # Evolución diaria durante el terreno - CORREGIDO
                df_evol = df_filtered.copy()
                df_evol["fecha"] = pd.to_datetime(df_evol["fecha"])
                df_evol = df_evol.sort_values("fecha")
                
                fig_evol = px.line(
                    df_evol,
                    x="fecha",
                    y="risk_score",
                    color="perfil",
                    markers=True,
                    title="Evolución del Score de Riesgo durante el Terreno"
                )
                st.plotly_chart(fig_evol, use_container_width=True)
            
            with col_graf2:
                # Distribución de perfiles
                perfil_counts = df_filtered["perfil"].value_counts()
                fig_pie = px.pie(
                    values=perfil_counts.values,
                    names=perfil_counts.index,
                    title="Distribución de Perfiles de Conducción",
                    color_discrete_map={
                        config.CLUSTER_NAMES[i]: config.CLUSTER_COLORS[i] 
                        for i in range(config.N_CLUSTERS)
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Tabla resumen por días
            st.markdown("### 📋 Resumen Diario del Terreno")
            
            # Preparar tabla de resumen - CORREGIDO
            df_display = df_filtered[[
                "fecha", "perfil", "distancia_km", "vel_media_mov", 
                "num_excesos", "risk_score", "harsh_accel_windows", "harsh_brake_windows"
            ]].copy()
            
            # CORRECCIÓN: Asegurar que fecha esté en formato datetime
            df_display["fecha"] = pd.to_datetime(df_display["fecha"])
            df_display["fecha_formato"] = df_display["fecha"].dt.strftime("%d/%m/%Y")
            
            cols_order = ["fecha_formato", "perfil", "distancia_km", "vel_media_mov", 
                         "num_excesos", "risk_score", "harsh_accel_windows", "harsh_brake_windows"]
            
            df_display_final = df_display[cols_order].rename(columns={
                "fecha_formato": "Fecha",
                "perfil": "Perfil",
                "distancia_km": "Distancia (km)",
                "vel_media_mov": "Vel. Media (km/h)",
                "num_excesos": "Excesos",
                "risk_score": "Score Riesgo",
                "harsh_accel_windows": "Acel. Bruscas",
                "harsh_brake_windows": "Fren. Bruscas"
            })
            
            st.dataframe(
                df_display_final.style.format({
                    "Distancia (km)": "{:.1f}",
                    "Vel. Media (km/h)": "{:.1f}",
                    "Score Riesgo": "{:.0f}"
                }),
                use_container_width=True,
                hide_index=True
            )
    
    with tab2:
        # TAB MODIFICADA: Análisis de día específico
        if 'selected_date' not in locals():
            selected_date = st.session_state.available_dates_in_range[0] if st.session_state.available_dates_in_range else None
        
        if selected_date:
            st.header(f"Análisis Detallado del {selected_date.strftime('%d/%m/%Y')}")
            
            # Obtener datos del día seleccionado
            day_metrics = st.session_state.df_metrics[
                st.session_state.df_metrics["fecha"] == selected_date
            ]
            
            if not day_metrics.empty:
                day_metrics = day_metrics.iloc[0]
                day_data = st.session_state.files_data[selected_date]
                df_day = day_data["df"]
                
                # Métricas del día vs promedio del terreno
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    delta_dist = day_metrics['distancia_km'] - df_filtered['distancia_km'].mean()
                    st.metric(
                        "Distancia del Día",
                        f"{day_metrics['distancia_km']:.1f} km",
                        delta=f"{delta_dist:+.1f} km vs terreno",
                        help="Comparado con el promedio del terreno"
                    )
                    
                with col2:
                    delta_vel = day_metrics['vel_media_mov'] - df_filtered['vel_media_mov'].mean()
                    st.metric(
                        "Velocidad Media",
                        f"{day_metrics['vel_media_mov']:.1f} km/h",
                        delta=f"{delta_vel:+.1f} km/h vs terreno"
                    )
                    
                with col3:
                    delta_exc = day_metrics['num_excesos'] - df_filtered['num_excesos'].mean()
                    st.metric(
                        "Excesos de Velocidad",
                        f"{day_metrics['num_excesos']}",
                        delta=f"{delta_exc:+.0f} vs terreno",
                        delta_color="inverse"
                    )
                    
                with col4:
                    if day_metrics['risk_score'] < 30:
                        risk_color = "🟢"
                    elif day_metrics['risk_score'] < 60:
                        risk_color = "🟡"
                    else:
                        risk_color = "🔴"
                        
                    delta_risk = day_metrics['risk_score'] - df_filtered['risk_score'].mean()
                    st.metric(
                        "Score de Riesgo",
                        f"{risk_color} {day_metrics['risk_score']:.0f}/100",
                        delta=f"{delta_risk:+.0f} vs terreno",
                        delta_color="inverse"
                    )
                
                # Perfil del día
                perfil_color = config.CLUSTER_COLORS[day_metrics['cluster']]
                st.markdown(
                    f"### Perfil del Día: "
                    f"<span style='color: {perfil_color}; font-weight: bold;'>"
                    f"{day_metrics['perfil']}</span>",
                    unsafe_allow_html=True
                )
                
                # Gráficos del día
                col_graf1, col_graf2 = st.columns(2)
                
                with col_graf1:
                    fig_radar = create_metrics_radar(day_metrics, st.session_state.cluster_centers)
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                with col_graf2:
                    fig_bars = create_metrics_bars(day_metrics, df_filtered)
                    st.plotly_chart(fig_bars, use_container_width=True)
                
                # Narrativa LLM
                st.markdown("### 🤖 Análisis Narrativo del Día (IA)")
                
                if st.session_state.openai_client:
                    if st.button("Generar Narrativa del Día", type="secondary"):
                        with st.spinner("Generando análisis..."):
                            narrative = generate_narrative(day_metrics, st.session_state.openai_client)
                            st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Ingresa tu API Key de OpenAI en la barra lateral para generar narrativas")
                
                # Timeline del día
                st.markdown("### 📈 Velocidad a lo largo del día")
                fig_timeline = create_daily_timeline(df_day)
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.error("No se encontraron datos para la fecha seleccionada")
    
    with tab3:
        # TAB SIN CAMBIOS MAYORES: Análisis de Clusters
        st.header("Análisis de Clusters del Terreno")
        
        cluster_analysis = analyze_cluster_characteristics(
            df_filtered, 
            st.session_state.cluster_centers
        )
        
        if cluster_analysis:
            cols_cluster = st.columns(len(cluster_analysis))
            for i, (cluster_id, analysis) in enumerate(cluster_analysis.items()):
                with cols_cluster[i]:
                    st.markdown(
                        f"<div style='background-color: {config.CLUSTER_COLORS[cluster_id]}20; "
                        f"border-left: 4px solid {config.CLUSTER_COLORS[cluster_id]}; "
                        f"padding: 10px; border-radius: 5px;'>"
                        f"<h4>{analysis['nombre']}</h4>"
                        f"<p><b>{analysis['num_dias']}</b> días</p>"
                        f"<p>Vel. media: <b>{analysis['caracteristicas']['vel_media_promedio']:.1f}</b> km/h</p>"
                        f"<p>Excesos prom: <b>{analysis['caracteristicas']['excesos_promedio']:.1f}</b></p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        
        # Scatter plot de clusters
        st.markdown("### Visualización de Clusters")
        
        col_x = st.selectbox(
            "Eje X", 
            ["vel_media_mov", "vel_max", "distancia_km", "num_excesos"],
            format_func=lambda x: {
                "vel_media_mov": "Velocidad Media",
                "vel_max": "Velocidad Máxima", 
                "distancia_km": "Distancia",
                "num_excesos": "Número de Excesos"
            }[x]
        )
        
        col_y = st.selectbox(
            "Eje Y",
            ["num_excesos", "excesos_por_km", "harsh_accel_windows", "risk_score"],
            format_func=lambda x: {
                "num_excesos": "Número de Excesos",
                "excesos_por_km": "Excesos por km",
                "harsh_accel_windows": "Aceleraciones Bruscas",
                "risk_score": "Score de Riesgo"
            }[x]
        )
        
        fig_scatter = create_cluster_scatter(df_filtered, col_x, col_y)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        # TAB MODIFICADA: Visualizaciones Avanzadas con validación
        st.header("Visualizaciones Avanzadas del Terreno")
        
        # Selector de día para visualizaciones específicas
        st.markdown("### 📅 Seleccionar Día para Visualizaciones Detalladas")
        
        if st.session_state.available_dates_in_range:
            viz_date = st.selectbox(
                "Día para visualizar GPS y detalles:",
                options=st.session_state.available_dates_in_range,
                format_func=lambda x: x.strftime("%d/%m/%Y"),
                help="Selecciona un día específico para visualizaciones detalladas",
                key="viz_date_selector"
            )
            
            if viz_date:
                st.markdown(f"**Día seleccionado:** {viz_date.strftime('%d/%m/%Y')}")
                
                # Heatmap de excesos por hora - SOLO DEL DÍA SELECCIONADO
                st.markdown("### 🔥 Distribución de Excesos por Hora del Día")
                
                if viz_date in st.session_state.files_data:
                    df_day_viz = st.session_state.files_data[viz_date]["df"]
                    
                    # Crear heatmap simplificado para un día
                    if "is_exceso" in df_day_viz.columns:
                        excesos_day = df_day_viz[df_day_viz["is_exceso"]]
                        
                        if not excesos_day.empty:
                            # Contar excesos por hora
                            excesos_por_hora = excesos_day["Hora"].dt.hour.value_counts().sort_index()
                            
                            if not excesos_por_hora.empty:
                                fig_hour_bar = px.bar(
                                    x=excesos_por_hora.index,
                                    y=excesos_por_hora.values,
                                    labels={'x': 'Hora del día', 'y': 'Número de excesos'},
                                    title=f"Excesos por hora - {viz_date.strftime('%d/%m/%Y')}"
                                )
                                fig_hour_bar.update_xaxis(tickmode='linear', tick0=0, dtick=1)
                                st.plotly_chart(fig_hour_bar, use_container_width=True)
                            else:
                                st.info("No se registraron excesos en este día")
                        else:
                            st.info("No se registraron excesos en este día")
                    else:
                        st.warning("No hay datos de excesos disponibles para este día")
                
                # Mapa del día seleccionado
                st.markdown(f"### 🗺️ Recorrido GPS del {viz_date.strftime('%d/%m/%Y')}")
                
                if viz_date in st.session_state.files_data:
                    df_map_day = st.session_state.files_data[viz_date]["df"]
                    fig_map, real_distance = create_map_view(df_map_day)
                    
                    if fig_map:
                        st.plotly_chart(fig_map, use_container_width=True)
                        
                        # Comparación de distancias
                        if real_distance:
                            day_metrics_map = st.session_state.df_metrics[
                                st.session_state.df_metrics["fecha"] == viz_date
                            ]
                            
                            if not day_metrics_map.empty:
                                day_metrics_map = day_metrics_map.iloc[0]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Distancia GPS", f"{real_distance:.1f} km")
                                with col2:
                                    st.metric("Distancia estimada", f"{day_metrics_map['distancia_km']:.1f} km")
                                with col3:
                                    diff = real_distance - day_metrics_map['distancia_km']
                                    st.metric("Diferencia", f"{diff:+.1f} km")
                    else:
                        st.info("No hay datos de GPS disponibles para este día")
                else:
                    st.error("No se encontraron datos para el día seleccionado")
                
                # Distribución de excesos por franja del día
                st.markdown(f"### 🕐 Excesos por Franja Horaria - {viz_date.strftime('%d/%m/%Y')}")
                
                day_metrics_viz = st.session_state.df_metrics[
                    st.session_state.df_metrics["fecha"] == viz_date
                ]
                
                if not day_metrics_viz.empty:
                    day_metrics_viz = day_metrics_viz.iloc[0]
                    if day_metrics_viz.get("excesos_por_franja"):
                        df_franjas_day = pd.DataFrame(
                            list(day_metrics_viz["excesos_por_franja"].items()),
                            columns=["Franja", "Cantidad"]
                        )
                        
                        if not df_franjas_day.empty and df_franjas_day["Cantidad"].sum() > 0:
                            fig_franjas_day = px.bar(
                                df_franjas_day, 
                                x="Franja", 
                                y="Cantidad",
                                color="Franja",
                                title=f"Excesos por franja horaria - {viz_date.strftime('%d/%m/%Y')}"
                            )
                            st.plotly_chart(fig_franjas_day, use_container_width=True)
                        else:
                            st.info("No hay excesos registrados por franjas para este día")
                    else:
                        st.info("No hay datos de excesos por franja para este día")
                
        else:
            st.warning("⚠️ Primero procesa los datos del vehículo para ver visualizaciones detalladas")
        
        # Heatmap general del terreno (opcional)
        st.markdown("---")
        st.markdown("### 🔥 Heatmap General del Terreno")
        
        if st.button("Generar Heatmap del Terreno Completo"):
            with st.spinner("Generando heatmap del terreno..."):
                fig_heatmap = create_hourly_heatmap(st.session_state.files_data)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.info("💡 Este heatmap muestra la distribución de excesos por hora para todos los días del terreno")
    
    with tab5:
        # TAB MODIFICADA: Análisis Comparativo
        st.header("Análisis Comparativo del Terreno")
        
        # Métricas estadísticas del terreno
        st.markdown("### 📊 Estadísticas del Terreno")
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            st.metric(
                "Promedio Diario",
                f"{df_filtered['distancia_km'].mean():.1f} km",
                help="Distancia promedio por día"
            )
            st.metric(
                "Desviación Estándar",
                f"{df_filtered['distancia_km'].std():.1f} km"
            )
        
        with stats_cols[1]:
            st.metric(
                "Día con Más Excesos",
                f"{df_filtered['num_excesos'].max()} excesos"
            )
            mejor_dia = df_filtered.loc[df_filtered['num_excesos'].idxmin(), 'fecha']
            st.metric(
                "Mejor Día",
                mejor_dia.strftime("%d/%m/%Y"),
                help="Día con menos excesos"
            )
        
        with stats_cols[2]:
            st.metric(
                "Score Promedio",
                f"{df_filtered['risk_score'].mean():.1f}/100"
            )
            st.metric(
                "Score Máximo",
                f"{df_filtered['risk_score'].max():.1f}/100"
            )
        
        with stats_cols[3]:
            dias_seguros = len(df_filtered[df_filtered['perfil'] == 'Conducción Segura'])
            porcentaje_seguro = (dias_seguros / len(df_filtered)) * 100
            st.metric(
                "Días Seguros",
                f"{dias_seguros}/{len(df_filtered)}"
            )
            st.metric(
                "% Conducción Segura",
                f"{porcentaje_seguro:.1f}%"
            )
        
        st.markdown("---")
        
        # Tabla comparativa detallada
        st.markdown("### 📋 Tabla Comparativa Detallada")
        
        # Preparar tabla con más detalles - CORREGIDO
        df_comparison = df_filtered[[
            "fecha", "perfil", "distancia_km", "vel_media_mov", "vel_max",
            "num_excesos", "excesos_por_km", "risk_score", 
            "harsh_accel_windows", "harsh_brake_windows"
        ]].copy()
        
        # CORRECCIÓN: Asegurar que fecha esté en formato datetime
        df_comparison["fecha"] = pd.to_datetime(df_comparison["fecha"])
        df_comparison = df_comparison.sort_values("fecha", ascending=False)
        df_comparison["fecha_formato"] = df_comparison["fecha"].dt.strftime("%d/%m/%Y")
        
        # Agregar ranking por score de riesgo
        df_comparison["ranking_riesgo"] = df_comparison["risk_score"].rank(ascending=False, method="min").astype(int)
        
        cols_comparison = [
            "fecha_formato", "ranking_riesgo", "perfil", "distancia_km", 
            "vel_media_mov", "vel_max", "num_excesos", "excesos_por_km", 
            "risk_score", "harsh_accel_windows", "harsh_brake_windows"
        ]
        
        df_comparison_final = df_comparison[cols_comparison].rename(columns={
            "fecha_formato": "Fecha",
            "ranking_riesgo": "Ranking Riesgo",
            "perfil": "Perfil",
            "distancia_km": "Distancia (km)",
            "vel_media_mov": "Vel. Media (km/h)",
            "vel_max": "Vel. Máx (km/h)",
            "num_excesos": "Excesos",
            "excesos_por_km": "Excesos/km",
            "risk_score": "Score Riesgo",
            "harsh_accel_windows": "Acel. Bruscas",
            "harsh_brake_windows": "Fren. Bruscas"
        })
        
        # Mostrar tabla con formato mejorado - SIN background_gradient
        st.dataframe(
            df_comparison_final.style.format({
                "Distancia (km)": "{:.1f}",
                "Vel. Media (km/h)": "{:.1f}",
                "Vel. Máx (km/h)": "{:.0f}",
                "Excesos/km": "{:.3f}",
                "Score Riesgo": "{:.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Análisis de tendencias
        st.markdown("### 📈 Análisis de Tendencias del Terreno")
        
        trend_metric = st.selectbox(
            "Métrica para análisis de tendencia:",
            ["risk_score", "num_excesos", "vel_media_mov", "distancia_km"],
            format_func=lambda x: {
                "risk_score": "Score de Riesgo",
                "num_excesos": "Número de Excesos",
                "vel_media_mov": "Velocidad Media",
                "distancia_km": "Distancia Diaria"
            }[x],
            key="trend_metric"
        )
        
        # Gráfico de tendencia SIN trendline (evitar statsmodels)
        df_filtered_sorted = df_filtered.copy()
        df_filtered_sorted["fecha"] = pd.to_datetime(df_filtered_sorted["fecha"])
        df_filtered_sorted = df_filtered_sorted.sort_values("fecha")
        
        fig_trend = px.scatter(
            df_filtered_sorted,
            x="fecha",
            y=trend_metric,
            color="perfil",
            # trendline="ols",  # REMOVIDO para evitar statsmodels
            title=f"Evolución de {trend_metric.replace('_', ' ').title()} durante el Terreno",
            color_discrete_map={
                config.CLUSTER_NAMES[i]: config.CLUSTER_COLORS[i] 
                for i in range(config.N_CLUSTERS)
            }
        )
        
        fig_trend.update_layout(
            xaxis_title="Fecha",
            yaxis_title=trend_metric.replace('_', ' ').title(),
            height=500
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Resumen ejecutivo del terreno
        st.markdown("### 📋 Resumen Ejecutivo del Terreno")
        
        if interval_summary:
            resumen_text = f"""
            **Análisis del Terreno - Vehículo {st.session_state.selected_vehicle}**
            
            **Período analizado:** {interval_summary['fecha_inicio'].strftime('%d/%m/%Y')} - {interval_summary['fecha_fin'].strftime('%d/%m/%Y')} ({interval_summary['num_dias']} días)
            
            **Métricas principales:**
            - Distancia total recorrida: **{interval_summary['distancia_total']:.1f} km**
            - Velocidad media del terreno: **{interval_summary['vel_media_promedio']:.1f} km/h**
            - Score de riesgo promedio: **{interval_summary['risk_score_promedio']:.1f}/100**
            - Total de excesos de velocidad: **{interval_summary['excesos_totales']}**
            
            **Perfil predominante:** {interval_summary['cluster_predominante']}
            
            **Comportamiento:**
            - Días con conducción segura: {dias_seguros}/{len(df_filtered)} ({porcentaje_seguro:.1f}%)
            - Velocidad máxima registrada: {interval_summary['vel_max_absoluta']:.0f} km/h
            - Tiempo total en excesos: {interval_summary['dur_exceso_total']:.0f} segundos
            """
            
            st.markdown(resumen_text)
            
            # Generar narrativa del terreno completo con IA
            if st.session_state.openai_client:
                st.markdown("### 🤖 Análisis Narrativo del Terreno Completo (IA)")
                
                if st.button("Generar Análisis del Terreno", type="primary"):
                    with st.spinner("Generando análisis del terreno completo..."):
                        
                        # Crear prompt para el terreno completo
                        prompt_terreno = f"""Analiza el siguiente comportamiento de conducción durante un terreno de trabajo:

Vehículo: {st.session_state.selected_vehicle}
Período: {interval_summary['fecha_inicio'].strftime('%d/%m/%Y')} - {interval_summary['fecha_fin'].strftime('%d/%m/%Y')} ({interval_summary['num_dias']} días)

RESUMEN DEL TERRENO:
- Distancia total: {interval_summary['distancia_total']:.1f} km
- Velocidad media: {interval_summary['vel_media_promedio']:.1f} km/h
- Velocidad máxima: {interval_summary['vel_max_absoluta']:.0f} km/h
- Excesos totales: {interval_summary['excesos_totales']}
- Score de riesgo promedio: {interval_summary['risk_score_promedio']:.1f}/100
- Perfil predominante: {interval_summary['cluster_predominante']}
- Días con conducción segura: {dias_seguros}/{len(df_filtered)} ({porcentaje_seguro:.1f}%)

Genera un análisis ejecutivo del comportamiento del conductor durante este terreno de trabajo (2-3 párrafos). 
Incluye recomendaciones específicas para mejorar la seguridad si es necesario.
Enfócate en el contexto laboral y la seguridad ocupacional."""

                        try:
                            response = st.session_state.openai_client.chat.completions.create(
                                model=config.LLM_MODEL,
                                messages=[{"role": "user", "content": prompt_terreno}],
                                temperature=config.LLM_TEMPERATURE,
                                max_tokens=config.LLM_MAX_TOKENS * 2  # Más tokens para análisis completo
                            )
                            narrative_terreno = response.choices[0].message.content.strip()
                            st.markdown(f'<div class="narrative-box">{narrative_terreno}</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generando narrativa del terreno: {str(e)}")
            else:
                st.warning("⚠️ Ingresa tu API Key de OpenAI para generar análisis narrativo del terreno")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "DriveTech Analytics Multi-Vehículo v2.0 | Sistema de Análisis de Conducción por Terrenos"
    "</div>",
    unsafe_allow_html=True
)