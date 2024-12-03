import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import numpy as np
import tensorflow as tf
import keras
import math  

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY], suppress_callback_exceptions=True)

# Data limpia
df=pd.read_csv("datos_limpios.csv", sep = ",")

df2 = df
df2['años'] = df["periodo"] // 10
df_promedio = df2.groupby("años", as_index=False).agg({"punt_global": "mean"})

# Crear la gráfica de líneas
fig_año = px.line(df_promedio, 
              x="años", 
              y="punt_global", 
              title="Promedio Global según el Año de Presentación",
              markers=True,
              labels={"punt_global": "Promedio Global", "años": "Año de Presentación"},
              color_discrete_sequence=["#28a745"])

fig_año.update_layout(
        plot_bgcolor="white",  # Fondo blanco
        title_font=dict(family="Arial", size=18, color='green', weight='bold'),  # Título en negrita
        title_x=0.5,  # Centra el título
    )

# Se trae el modelo, para poder predecir
model = keras.models.load_model('modelo_proyecto3.keras')

# Layout de la pestaña de inicio del dash
home_layout = html.Div([
    html.H1("Predice el Resultado del ICFES Saber 11 en Norte de Santander",className = "text-success",  style={'textAlign': 'center'}),
    html.H4("Bienvenido a la Plataforma de Análisis de los resultados del Saber 11!",className = "text-success", style={'textAlign': 'center'}),
    # Texto explicatorio del tablero
    dbc.Card([
        dbc.CardBody([
            html.P(
                """En esta aplicación, podrás explorar los resultados de un estudio realizado por un grupo de estudiantes de la clase de analítica computacional 
                para la toma de decisiones. Hemos diseñado esta plataforma para que puedas acceder a un análisis detallado de los 
                resultados del ICFES en el Norte de Santander, permitiéndote comprender mejor las variables que influyen en el desempeño académico de los estudiantes en la región.
                
                Además, tendrás la oportunidad de ingresar información de un nuevo estudiante en nuestra sección de predicciones. A través de este proceso, 
                podrás evaluar su posible desempeño en el examen ICFES con base en las tendencias observadas en los datos existentes.

                Te invitamos a navegar por las diferentes secciones de la aplicación y descubrir todo lo que tenemos para ofrecerte.
                """,
                className = "text-info", 
                style={
                    'textAlign': 'justify',
                    'margin': '0 auto',
                    'maxWidth': '1100px',
                    'padding': '10px'
                }
            ),
        ])
    ], className="mt-4 border-info", style={'borderWidth': '2px', 'borderStyle': 'solid', 'marginTop': '5px'}),
    dbc.Card([
        dbc.CardBody([
            html.H5("Trabajo realizado por:",className = "text-info", style={'textAlign': 'center'}),
            html.P("- Analista de negocios y Diseñador del tablero: Samuel Pedroza ",className = "text-info", style={'textAlign': 'center'}),
            html.P("- Ingeniero de datos y Científico de datos: Jorge Russi ",className = "text-info", style={'textAlign': 'center'}),
            html.P("- Analista de datos y Encargado del despliegue: Cristian Rincón",className = "text-info", style={'textAlign': 'center'})
        ])
    ], className="mt-4 border-info", style={'borderWidth': '2px', 'borderStyle': 'solid'}),
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('Mapa_de_Norte_de_Santander.png'), style={'width': '10%', 'margin': '1px'}),
            html.Img(src=app.get_asset_url('ICFES.png'), style={'width': '35%', 'margin': '1px'})
        ], style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '10px'})
    ])
])

# Layout de la pestaña de exploracion de data del dash
data_exploration_layout = html.Div([
    html.H1("Exploración de data", className = "text-success", style={'textAlign': 'center'}),
    html.H3("""¿Te has preguntado alguna vez qué podría estar detrás de los resultados del Saber 11? En este espacio encontrarás gráficos que nos ayudan a entender 
            cómo diferentes factores pueden influir en el desempeño académico. La información está dividida en tres bloques clave, diseñados para que puedas explorar 
            estas conexiones y reflexionar sobre ellas a tu propio ritmo. Aquí te dejo todo listo para que plantees tus propias preguntas y descubras nuevas perspectivas:""", 
            className = "text-info",
            style={'textAlign': 'justify',
                    'margin': '0 auto',
                    'maxWidth': '1500px',
                    'padding': '20px'}),
    html.H4("¿Cómo influyen los factores familiares en los resultados del Saber 11?",className = "text-info", style={'textAlign': 'center'}),
    html.H4("¿Las características del colegio tienen un impacto significativo en el puntaje global de los estudiantes?",className = "text-info", style={'textAlign': 'center'}),
    html.H4("¿Importan los factores temporales y espaciales en los resultados del Saber 11?",className = "text-info", style={'textAlign': 'center'}),

    # Primera caja con los grafiicos de informacion personal del cliente
    html.H2("Factores Familiares:", className="text-success", style={'paddingLeft': '20px', 'paddingRight': '20px'}),
    dbc.Card([
        dbc.CardBody([
            html.Label("Selecciona una variable:",className="form-label text-info", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='variable_dropdown_graf1',
                options=[
                    {'label': 'Cuartos por hogar', 'value': 'fami_cuartoshogar'},
                    {'label': 'Educación de la madre', 'value': 'fami_educacionmadre'},
                    {'label': 'Educación del padre', 'value': 'fami_educacionpadre'},
                    {'label': 'Estrato de la vivienda', 'value': 'fami_estratovivienda'},
                    {'label': 'Personas por hogar', 'value': 'fami_personashogar'},
                    {'label': 'Automóvil', 'value': 'fami_tieneautomovil'},
                    {'label': 'Computador', 'value': 'fami_tienecomputador'},
                    {'label': 'Internet', 'value': 'fami_tieneinternet'},
                    {'label': 'Lavadora', 'value': 'fami_tienelavadora'},
                ],
                value="fami_cuartoshogar",  # Valor inicial
                style={'marginBottom': '15px'},
                className="text-info"
            ),
            
            # Gráfico dinámico
            dcc.Graph(id='dynamic-graph1')
        ])
    ], className="mt-4 border-info", style={'borderWidth': '2px', 'borderStyle': 'solid', 'marginTop': '5px'}),
    html.Div(style={'height': '30px'}),
    # Segunda caja con la informacion relacionada con el ultimo contacto con el cliente
    html.H2("Características del Colegio:", className="text-success", style={'paddingLeft': '20px', 'paddingRight': '20px'}),
    dbc.Card([
        dbc.CardBody([
            html.Label("Selecciona una variable:",className="form-label text-info", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='variable_dropdown_graf2',
                options=[
                    {'label': 'Área de ubicación', 'value': 'cole_area_ubicacion'},
                    {'label': 'Tipo de formación', 'value': 'cole_caracter'},
                    {'label': 'Géneros en el colegio', 'value': 'cole_genero'},
                    {'label': 'Tipos de jornada en el colegio', 'value': 'cole_jornada'},
                    {'label': 'Naturaleza del colegio', 'value': 'cole_naturaleza'},
                ],
                value="cole_area_ubicacion",  # Valor inicial
                style={'marginBottom': '15px'},
                className="text-info"
            ),
            
            # Gráfico dinámico
            dcc.Graph(id='dynamic-graph2')
        ])
    ], className="mt-4 border-info", style={'borderWidth': '2px', 'borderStyle': 'solid', 'marginTop': '5px'}),
    html.Div(style={'height': '30px'}),

    # Tercer caja con la informacion adicional de la ultima campaña
    html.H2("Factores temporales y espaciales:", className="text-success", style={'paddingLeft': '20px', 'paddingRight': '20px'}),
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Promedio global segúne el año de presentación:",className="form-label text-info", style={'fontWeight': 'bold'}),
                    dcc.Graph(figure=fig_año)
                ], width=7),

                dbc.Col([
                    html.Label("Mejores y peores municipios por promedio global", 
                        className="form-label text-info", 
                        style={'fontWeight': 'bold'}),
                    # Botones debajo del label
                    html.Div([
                        html.Button("10 Mejores", id="btn-top-10", n_clicks=0, className="btn btn-success me-2"),
                        html.Button("10 Peores", id="btn-bottom-10", n_clicks=0, className="btn btn-danger")
                            ], style={"display": "flex", "flexDirection": "row", "alignItems": "center"}),
                        dcc.Graph(id="bar-graph")
                        ], width=5),
            ], className="mb-4"),
        ])
    ], className="mt-4 border-info", style={'borderWidth': '2px', 'borderStyle': 'solid', 'marginTop': '5px'}),
])

# Callback para actualizar la gráfica
@app.callback(
    Output('dynamic-graph1', 'figure'),
    [Input('variable_dropdown_graf1', 'value')]
)
def update_graph(selected_variable):
    
    grouped_df = df.groupby(selected_variable)['punt_global'].mean().reset_index()
    
    # Crear la gráfica de barras con un color verde (success)
    fig = px.bar(
        grouped_df,
        x=selected_variable,  # Usamos la columna seleccionada como eje X
        y='punt_global',  # Mostramos el promedio de punt_global en el eje Y
        title=f'Promedio de Puntaje Global por {selected_variable}',
        labels={selected_variable: selected_variable, 'punt_global': 'Promedio de Puntaje Global'},
        color_discrete_sequence=["#28a745"]  # Estilo 'success' en color verde
    )
    
    # Personalizar el diseño de la gráfica
    fig.update_layout(
        plot_bgcolor="white",  # Fondo blanco
        title_font=dict(family="Arial", size=18, color='green', weight='bold'),  # Título en negrita
        title_x=0.5,  # Centra el título
    )
    
    return fig

# Callback para actualizar la gráfica
@app.callback(
    Output('dynamic-graph2', 'figure'),
    [Input('variable_dropdown_graf2', 'value')]
)
def update_graph(selected_variable):
    
    grouped_df = df.groupby(selected_variable)['punt_global'].mean().reset_index()
    
    # Crear la gráfica de barras con un color verde (success)
    fig = px.bar(
        grouped_df,
        x=selected_variable,  # Usamos la columna seleccionada como eje X
        y='punt_global',  # Mostramos el promedio de punt_global en el eje Y
        title=f'Promedio de Puntaje Global por {selected_variable}',
        labels={selected_variable: selected_variable, 'punt_global': 'Promedio de Puntaje Global'},
        color_discrete_sequence=["#28a745"]  # Estilo 'success' en color verde
    )
    
    # Personalizar el diseño de la gráfica
    fig.update_layout(
        plot_bgcolor="white",  # Fondo blanco
        title_font=dict(family="Arial", size=18, color='green', weight='bold'),  # Título en negrita
        title_x=0.5,  # Centra el título
    )
    
    return fig

promedios_por_muni = df.groupby('cole_mcpio_ubicacion')['punt_global'].mean().reset_index()

# Callback para actualizar la gráfica de barras de mejores/peores municipios
@app.callback(
    Output("bar-graph", "figure"),
    [Input("btn-top-10", "n_clicks"),
     Input("btn-bottom-10", "n_clicks")]
)
def update_bar_graph(n_clicks_top, n_clicks_bottom):
    # Determinar qué botón fue presionado
    ctx = dash.callback_context
    if not ctx.triggered:
        # Si no se ha hecho clic en ningún botón, no se muestra nada
        return {
            "data": [],
            "layout": {"title": "Por favor selecciona un botón"}
        }

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "btn-top-10":
        # Ordenar los municipios por el promedio del resultado de manera descendente
        df_sorted = promedios_por_muni.sort_values("punt_global", ascending=False).head(10)
        title = "10 Mejores Municipios"
    elif button_id == "btn-bottom-10":
        # Ordenar los municipios por el promedio del resultado de manera ascendente
        df_sorted = promedios_por_muni.sort_values("punt_global", ascending=True).head(10)
        title = "10 Peores Municipios"
    else:
        df_sorted = promedios_por_muni
        title = "punt_global"

    # Crear la gráfica de barras
    fig_bar = px.bar(df_sorted, 
                     x="cole_mcpio_ubicacion", 
                     y="punt_global", 
                     title=title,
                     labels={"punt_global": "Promedio Global", "cole_mcpio_ubicacion": "Municipio"},
                     color_discrete_sequence=["#28a745"])
    
    fig_bar.update_layout(
        plot_bgcolor="white",  # Fondo blanco
        title_font=dict(family="Arial", size=18, color='green', weight='bold'),  # Título en negrita
        title_x=0.5,  # Centra el título,
    )

    return fig_bar


texto = 'cole_mcpio_ubicacion_'
mi_diccionario_mun_col = {ubicacion: f"{texto}{ubicacion}" for ubicacion in sorted(df["cole_mcpio_ubicacion"].unique())}
dropdown_options_mun_col = [{'label': key, 'value': value} for key, value in mi_diccionario_mun_col.items()]

texto = 'estu_mcpio_presentacion_'
mi_diccionario_mun_pres = {ubicacion: f"{texto}{ubicacion}" for ubicacion in sorted(df["estu_mcpio_presentacion"].unique())}
dropdown_options_mun_pres = [{'label': key, 'value': value} for key, value in mi_diccionario_mun_pres.items()]

texto = 'estu_mcpio_reside_'
mi_diccionario_mun_resid = {ubicacion: f"{texto}{ubicacion}" for ubicacion in sorted(df["estu_mcpio_reside"].unique())}
dropdown_options_mun_resid = [{'label': key, 'value': value} for key, value in mi_diccionario_mun_resid.items()]

texto = 'fami_educacionmadre_'
mi_diccionario_edu_madre = {ubicacion: f"{texto}{ubicacion}" for ubicacion in sorted(df["fami_educacionmadre"].unique())}
dropdown_options_edu_madre = [{'label': key, 'value': value} for key, value in mi_diccionario_edu_madre.items()]

texto = 'fami_educacionpadre_'
mi_diccionario_edu_padre = {ubicacion: f"{texto}{ubicacion}" for ubicacion in sorted(df["fami_educacionpadre"].unique())}
dropdown_options_edu_padre = [{'label': key, 'value': value} for key, value in mi_diccionario_edu_padre.items()]

texto = 'fami_cuartoshogar_'
mi_diccionario_cuarto_hogar = {ubicacion: f"{texto}{ubicacion}" for ubicacion in sorted(df["fami_cuartoshogar"].unique())}
dropdown_options_cuarto_hogar = [{'label': key, 'value': value} for key, value in mi_diccionario_cuarto_hogar.items()]

texto = 'fami_personashogar_'
mi_diccionario_personas_hogar = {ubicacion: f"{texto}{ubicacion}" for ubicacion in sorted(df["fami_personashogar"].unique())}
dropdown_options_personas_hogar = [{'label': key, 'value': value} for key, value in mi_diccionario_personas_hogar.items()]

texto = 'fami_estratovivienda_'
mi_diccionario_estrato = {ubicacion: f"{texto}{ubicacion}" for ubicacion in sorted(df["fami_estratovivienda"].unique())}
dropdown_options_estrato = [{'label': key, 'value': value} for key, value in mi_diccionario_estrato.items()]

texto = 'cole_jornada_'
mi_diccionario_jornada = {ubicacion: f"{texto}{ubicacion}" for ubicacion in sorted(df["cole_jornada"].unique())}
dropdown_options_jornada = [{'label': key, 'value': value} for key, value in mi_diccionario_jornada.items()]

texto = 'cole_caracter_'
mi_diccionario_formacion = {ubicacion: f"{texto}{ubicacion}" for ubicacion in sorted(df["cole_caracter"].unique())}
dropdown_options_formacion = [{'label': key, 'value': value} for key, value in mi_diccionario_formacion.items()]

texto = 'desemp_ingles_'
mi_diccionario_ingles_est = {ubicacion: f"{texto}{ubicacion}" for ubicacion in sorted(df["desemp_ingles"].unique())}
dropdown_options_ingles_est = [{'label': key, 'value': value} for key, value in mi_diccionario_ingles_est.items()]


# Layout de la pestaña de predicciones del dash
predictions_layout = html.Div([
    html.H1("Predicciones", className="text-success", style={'textAlign': 'center'}),
    html.H3("Ingrese los datos del estudiante:", className="text-info", style={'textAlign': 'center'}),
    # Primer caja, con el primer set de inputs
    html.H4("Información de aspectos familiares y económicos:", className="text-success", style={'paddingLeft': '20px', 'paddingRight': '20px'}),
    dbc.Card([
        dbc.CardBody([
            # Linea numero 1 de la caja
            dbc.Row([
                dbc.Col([
                    html.Label("¿Tiene automóvil?", className="form-label text-info",),
                    dbc.RadioItems(
                        id='automovil-options',
                        options=[
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' Si'], className="d-flex align-items-center"),
                            'value': 'fami_tieneautomovil_Si'},
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' No'], className="d-flex align-items-center"),
                            'value': 'fami_tieneautomovil_No'},
                        ],
                        value=None,
                        labelClassName="form-check-label",
                        inputClassName="form-check-input",
                        style={'display': 'flex', 'flex-direction': 'column'}
                    )
                ], width=3),
               dbc.Col([
                    html.Label("¿Tiene computador?", className="form-label text-info",),
                    dbc.RadioItems(
                        id='computador-options',
                        options=[
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' Si'], className="d-flex align-items-center"),
                            'value': 'fami_tienecomputador_Si'},
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' No'], className="d-flex align-items-center"),
                            'value': 'fami_tienecomputador_No'},
                        ],
                        value=None,
                        labelClassName="form-check-label",
                        inputClassName="form-check-input",
                        style={'display': 'flex', 'flex-direction': 'column'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("¿Tiene internet?", className="form-label text-info",),
                    dbc.RadioItems(
                        id='internet-options',
                        options=[
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' Si'], className="d-flex align-items-center"),
                            'value': 'fami_tieneinternet_Si'},
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' No'], className="d-flex align-items-center"),
                            'value': 'fami_tieneinternet_No'},
                        ],
                        value=None,
                        labelClassName="form-check-label",
                        inputClassName="form-check-input",
                        style={'display': 'flex', 'flex-direction': 'column'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("¿Tiene lavadora?", className="form-label text-info",),
                    dbc.RadioItems(
                        id='lavadora-options',
                        options=[
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' Si'], className="d-flex align-items-center"),
                            'value': 'fami_tienelavadora_Si'},
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' No'], className="d-flex align-items-center"),
                            'value': 'fami_tienelavadora_No'},
                        ],
                        value=None,
                        labelClassName="form-check-label",
                        inputClassName="form-check-input",
                        style={'display': 'flex', 'flex-direction': 'column'}
                    )
                ], width=3)
            ], className="mb-4", justify='between', style={'marginBottom': '30px'}),
            dbc.Row([
                dbc.Col([
                    html.Label("Nivel de eduación de la madre", className="form-label text-info",),
                    dcc.Dropdown(
                        id="educ_madre-dropdown",
                        options=dropdown_options_edu_madre,
                        value=None,
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Nivel de eduación del padre", className="form-label text-info",),
                    dcc.Dropdown(
                        id="educ_padre-dropdown",
                        options=dropdown_options_edu_padre,
                        value=None,
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Cuartos en el hogar", className="form-label text-info",),
                    dcc.Dropdown(
                        id="cuartos_hogar-dropdown",
                        options=dropdown_options_cuarto_hogar,
                        value=None,
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Personas en el hogar", className="form-label text-info",),
                    dcc.Dropdown(
                        id="personas_hogar-dropdown",
                        options=dropdown_options_personas_hogar,
                        value=None,
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Estrato de la vivienda", className="form-label text-info",),
                    dcc.Dropdown(
                        id="estrato-dropdown",
                        options=dropdown_options_estrato,
                        value=None,
                    )
                ], width=2)
            ], className="mb-4 border-info" , justify='between', style={'marginBottom': '30px'})
        ])
    ], className="mt-4 border-info", style={'borderWidth': '2px', 'borderStyle': 'solid'}),
    
    # Barra de progreso
    dbc.Progress(id="progress-bar1", value=0, className="mt-3", color="success"),
    html.Div(style={'height': '30px'}),

    # Segunda caja, con el segundo set de inputs
    html.H4("Información geográfica:", className="text-success", style={'paddingLeft': '20px', 'paddingRight': '20px'}),
    dbc.Card([
        dbc.CardBody([
            # Unica linea de esta caja, tiene day, duration, month y contact
            dbc.Row([
                dbc.Col([
                    html.Label("Municipio en el que se encuentra el colegio", className="form-label text-info",),
                    dcc.Dropdown(
                        id="mun_colegio-dropdown",
                        options=dropdown_options_mun_col,
                        value=None,
                    )
                ], width=4),

                dbc.Col([
                    html.Label("Municipio en el que el estudiante presenta la prueba", className="form-label text-info",),
                    dcc.Dropdown(
                        id="mun_presentacion-dropdown",
                        options=dropdown_options_mun_pres,
                        value=None,
                    )
                ], width=4),

                dbc.Col([
                    html.Label("Municipio en el que vive el estudiante", className="form-label text-info",),
                    dcc.Dropdown(
                        id="mun_residencia-dropdown",
                        options=dropdown_options_mun_resid,
                        value=None,
                    )
                ], width=4)
            ])
        ])
    ], className="mt-4 border-info", style={'borderWidth': '2px', 'borderStyle': 'solid'}),

    # Barra de progreso
    dbc.Progress(id="progress-bar2", value=0, className="mt-3", color="success"),
    html.Div(style={'height': '30px'}),

    # Tercer caja, con el tercer set de inputs
    html.H4(" Información sobre el colegio:", className="text-success", style={'paddingLeft': '20px', 'paddingRight': '20px'}),
    dbc.Card([
        dbc.CardBody([
            # Unica linea de esta caja, tiene poutcome, campain, pdays, previous
            dbc.Row([
                dbc.Col([
                    html.Label("¿Área de ubicación?", className="form-label text-info",),
                    dbc.RadioItems(
                        id='area_ubc-options',
                        options=[
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' Rural'], className="d-flex align-items-center"),
                            'value': 'cole_area_ubicacion_RURAL'},
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' Urbano'], className="d-flex align-items-center"),
                            'value': 'cole_area_ubicacion_URBANO'},
                        ],
                        value=None,
                        labelClassName="form-check-label",
                        inputClassName="form-check-input",
                        style={'display': 'flex', 'flex-direction': 'column'}
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Naturaleza del colegio", className="form-label text-info",),
                    dbc.RadioItems(
                        id='naturaleza-options',
                        options=[
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' Oficial'], className="d-flex align-items-center"),
                            'value': 'cole_naturaleza_OFICIAL'},
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' No Oficial'], className="d-flex align-items-center"),
                            'value': 'cole_naturaleza_NO OFICIAL'},
                        ],
                        value=None,
                        labelClassName="form-check-label",
                        inputClassName="form-check-input",
                        style={'display': 'flex', 'flex-direction': 'column'}
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Géneros en el colegio", className="form-label text-info",),
                    dbc.RadioItems(
                        id='genero_colegio-options',
                        options=[
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' Femenino'], className="d-flex align-items-center"),
                            'value': 'cole_genero_FEMENINO'},
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' Masculino'], className="d-flex align-items-center"),
                            'value': 'cole_genero_MASCULINO'},
                            {'label': html.Div([html.I(className="fas fa-circle-o"), ' Mixto'], className="d-flex align-items-center"),
                            'value': 'cole_genero_MIXTO'},
                        ],
                        value=None,
                        labelClassName="form-check-label",
                        inputClassName="form-check-input",
                        style={'display': 'flex', 'flex-direction': 'column'}
                    )
                ], width=2),
                dbc.Col([
                    dbc.Row([
                        html.Label("Tipo de formación", className="form-label text-info",),
                            dcc.Dropdown(
                                id="formacion-dropdown",
                                options=dropdown_options_formacion,
                                value=None,
                                        ),
                            ], className="mb-3"),
                    dbc.Row([
                        html.Label("Tipo de jornada", className="form-label text-info",),
                            dcc.Dropdown(
                                id="jornada-dropdown",
                                options=dropdown_options_jornada,
                                value=None,
                                        ),
                    ]),
                ], width=6)
            ])
        ])
    ], className="mt-4 border-info", style={'borderWidth': '2px', 'borderStyle': 'solid'}),

    # Barra de progreso
    dbc.Progress(id="progress-bar3", value=0, className="mt-3", color="success"),
    html.Div(style={'height': '30px'}),

    # Cuarta caja, con el segundo set de inputs
    html.H4("Información del estudiante:", className="text-success", style={'paddingLeft': '20px', 'paddingRight': '20px'}),
    dbc.Card([
        dbc.CardBody([
            # Unica linea de esta caja, tiene day, duration, month y contact
            dbc.Row([
                dbc.Col([
                    html.Label("Género del estudiante", className="form-label text-info",),
                    dcc.Dropdown(
                        id="genero_est-dropdown",
                        options=[
                    {'label': 'Masculino', 'value': 'estu_genero_M'},
                    {'label': 'Femenino', 'value': 'estu_genero_F'},
                ],
                        value=None,
                    )
                ], width=6),

                dbc.Col([
                    html.Label("Nivel de inglés", className="form-label text-info",),
                    dcc.Dropdown(
                        id="ingles_est-dropdown",
                        options=dropdown_options_ingles_est,
                        value=None,
                    )
                ], width=6),
            ])
        ])
    ], className="mt-4 border-info", style={'borderWidth': '2px', 'borderStyle': 'solid'}),

    # Barra de progreso
    dbc.Progress(id="progress-bar4", value=0, className="mt-3", color="success"),



    # Aviso en forma de warning siempre
    dbc.Alert(
        [
            html.H4("¡Advertencia!", className="alert-heading"),  # Alert heading
            html.P("Por favor, complete todos los campos antes de hacer la predicción. Asegúrese de que todos los valores sean válidos.", className="mb-0"),
            html.P("Si tienes dudas, consulta la documentación o contáctanos.", className="mb-0")
        ],
        color="warning",
        is_open=True, 
        dismissable=True,
    ),
    
    # Aviso porque si no estan bien los inputs
    html.Div(id='warning-message', className="text-danger", style={'marginTop': '10px'}),

    # Boton que corre las predicciones
    html.Div(className="d-grid gap-2", style={'marginTop': '20px'}, children=[
        dbc.Button("Resultado Esperado Visible", id="predict-button", className="btn btn-lg btn-primary", n_clicks=0),

    html.Div(
    id='prediction-output',
    className="text-success",
    style={
        'marginTop': '20px',
        'fontSize': '50px',  # Tamaño de la fuente grande
        'textAlign': 'center',  # Centra el texto horizontalmente
        'display': 'flex',  # Usamos flexbox para centrar el contenido
        'justifyContent': 'center',  # Centra horizontalmente en el contenedor
        'alignItems': 'center',  # Centra verticalmente en el contenedor
        'height': '100px'  # Asegura que haya espacio suficiente para el contenido
    }
        )


    ])
])

@app.callback(
    Output('progress-bar1', 'value'),
    Output('progress-bar2', 'value'),
    Output('progress-bar3', 'value'),
    Output('progress-bar4', 'value'),
    Input('automovil-options', 'value'),
    Input('computador-options', 'value'),
    Input('internet-options', 'value'),
    Input('lavadora-options', 'value'),
    Input('educ_madre-dropdown', 'value'),
    Input('educ_padre-dropdown', 'value'),
    Input('cuartos_hogar-dropdown', 'value'),
    Input('personas_hogar-dropdown', 'value'),
    Input('estrato-dropdown', 'value'),
    Input('mun_colegio-dropdown', 'value'),
    Input('mun_presentacion-dropdown', 'value'),
    Input('mun_residencia-dropdown', 'value'),
    Input('area_ubc-options', 'value'),
    Input('naturaleza-options', 'value'),
    Input('genero_colegio-options', 'value'),
    Input('formacion-dropdown', 'value'),
    Input('jornada-dropdown', 'value'),
    Input('genero_est-dropdown', 'value'),
    Input('ingles_est-dropdown', 'value')
)
def update_progress(automovil_value, computador_value, internet_value, lavadora_value, educ_madre_value, 
                    educ_padre_value, cuartos_hogar_value, personas_hogar_value, estrato_value, mun_colegio_value, 
                    mun_presentacion_value, mun_residencia_value, area_ubc_value, naturaleza_value, genero_colegio_value, 
                    formacion_value, jornada_value, genero_est_value, ingles_est_value):
    total_inputs1 = 9
    filled_inputs1 = sum([
        automovil_value is not None, 
        computador_value is not None, 
        internet_value is not None, 
        lavadora_value is not None, 
        educ_madre_value is not None, 
        educ_padre_value is not None, 
        cuartos_hogar_value is not None, 
        personas_hogar_value is not None,
        estrato_value is not None
    ])
    progress_percentage1 = (filled_inputs1 / total_inputs1) * 100

    total_inputs2 = 3
    filled_inputs2 = sum([
        mun_colegio_value is not None,  
        mun_presentacion_value is not None, 
        mun_residencia_value is not None, 
    ])
    progress_percentage2 = (filled_inputs2 / total_inputs2) * 100

    total_inputs3 = 5
    filled_inputs3 = sum([
        area_ubc_value is not None,  
        naturaleza_value is not None, 
        genero_colegio_value is not None,  
        formacion_value is not None, 
        jornada_value is not None,
    ])
    progress_percentage3 = (filled_inputs3 / total_inputs3) * 100

    total_inputs4 = 2
    filled_inputs4 = sum([
        genero_est_value is not None,  
        ingles_est_value is not None, 
    ])
    progress_percentage4 = (filled_inputs4 / total_inputs4) * 100


    return progress_percentage1, progress_percentage2, progress_percentage3, progress_percentage4

data_x = pd.read_csv("datax.csv")

@app.callback(
    Output('warning-message', 'children'),
    Output('prediction-output', 'children'),
    Input('automovil-options', 'value'),
    Input('computador-options', 'value'),
    Input('internet-options', 'value'),
    Input('lavadora-options', 'value'),
    Input('educ_madre-dropdown', 'value'),
    Input('educ_padre-dropdown', 'value'),
    Input('cuartos_hogar-dropdown', 'value'),
    Input('personas_hogar-dropdown', 'value'),
    Input('estrato-dropdown', 'value'),
    Input('mun_colegio-dropdown', 'value'),
    Input('mun_presentacion-dropdown', 'value'),
    Input('mun_residencia-dropdown', 'value'),
    Input('area_ubc-options', 'value'),
    Input('naturaleza-options', 'value'),
    Input('genero_colegio-options', 'value'),
    Input('formacion-dropdown', 'value'),
    Input('jornada-dropdown', 'value'),
    Input('genero_est-dropdown', 'value'),
    Input('ingles_est-dropdown', 'value'),
    Input('predict-button', 'n_clicks')
)
def on_predict(automovil_value, computador_value, internet_value, lavadora_value, educ_madre_value, 
                    educ_padre_value, cuartos_hogar_value, personas_hogar_value, estrato_value, mun_colegio_value, 
                    mun_presentacion_value, mun_residencia_value, area_ubc_value, naturaleza_value, genero_colegio_value, 
                    formacion_value, jornada_value, genero_est_value, ingles_est_value, n_clicks):
    if n_clicks > 0:  # Only run if the button was clicked
        # Check if all inputs are filled
        inputs_filled = all([
            automovil_value is not None, 
            computador_value is not None, 
            internet_value is not None, 
            lavadora_value is not None, 
            educ_madre_value is not None, 
            educ_padre_value is not None, 
            cuartos_hogar_value is not None, 
            personas_hogar_value is not None,
            estrato_value is not None,
            mun_colegio_value is not None,  
            mun_presentacion_value is not None, 
            mun_residencia_value is not None,
            area_ubc_value is not None,  
            naturaleza_value is not None, 
            genero_colegio_value is not None,  
            formacion_value is not None, 
            jornada_value is not None,
            genero_est_value is not None,  
            ingles_est_value is not None,
        ])
        if inputs_filled:
            variables = {
                "area_ubicación": list(data_x.columns[0:1]),

                "cole_formacion": list(data_x.columns[1:4]),

                "cole_genero": list(data_x.columns[4:6]),

                "cole_jornada": list(data_x.columns[6:11]),

                "cole_muni_ubic": list(data_x.columns[11:49]),

                "cole_naturaleza": list(data_x.columns[49:50]),

                "estudiante_genero": list(data_x.columns[50:51]),

                "estudiante_municipio_pres": list(data_x.columns[51:64]),

                "estudiante_municipio_resd": list(data_x.columns[64:103]),

                "cuartos_por_hogar": list(data_x.columns[103:108]),

                "educacion_madre": list(data_x.columns[108:119]),

                "educacion_padre": list(data_x.columns[119:130]),

                "estrato": list(data_x.columns[130:136]),

                "personas_por_hogar": list(data_x.columns[136:140]),

                "auto": list(data_x.columns[140:141]),

                "computador": list(data_x.columns[141:142]),

                "internet": list(data_x.columns[142:143]),

                "lavadora": list(data_x.columns[143:144]),

                "nivel_ingles": list(data_x.columns[144:]),
            }
            x = []
            dropdown_values = {
                "auto": automovil_value,
                "computador": computador_value,
                "internet": internet_value,
                "lavadora": lavadora_value,
                "educacion_madre": educ_madre_value,
                "educacion_padre": educ_padre_value,
                "cuartos_por_hogar": cuartos_hogar_value,
                "personas_por_hogar": personas_hogar_value,
                "estrato": estrato_value,
                "cole_muni_ubic": mun_colegio_value,
                "estudiante_municipio_pres": mun_presentacion_value,
                "estudiante_municipio_resd": mun_residencia_value,
                "area_ubicación": area_ubc_value,
                "cole_naturaleza": naturaleza_value,
                "cole_genero": genero_colegio_value,
                "cole_formacion": formacion_value,
                "cole_jornada": jornada_value,
                "estudiante_genero": genero_est_value,  
                "nivel_ingles": ingles_est_value,
            }

            for var, options in variables.items():
                selected_value = dropdown_values[var]
                x.extend([1 if option == selected_value else 0 for option in options])

            prediction_result = model.predict([x]) 
            resultado = prediction_result[0][0]

            prediction_message = f"El resultado esperado es: {resultado:.0f}"

            return None, prediction_message
        else:
            return dbc.Alert(
                html.Div([
                    html.H4("¡Error!", className="alert-heading"),
                    html.P("Por favor, ¡complete todos los campos! Esto lo puede hacer asegurandose que las 4 barras esten completas", className="mb-0"),  # Custom warning message
                ]),    
                color="danger", 
                is_open=True,
                dismissable=True,
                duration=7000, 
            ), None
    
    return dash.no_update  # If button has not been clicked, do nothing

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Detectar la URL activa
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Inicio", href="/home", id="home-link")),
            dbc.NavItem(dbc.NavLink("Exploración de data", href="/data-exploration", id="data-exploration-link")),
            dbc.NavItem(dbc.NavLink("Predicciones", href="/predictions", id="predictions-link")),
        ],
        brand="Saber 11 Analytics",
        brand_href="#",
        color="success",
        dark=True,
        expand="lg",
        className="mb-4",
        brand_style={
        'font-size': '28px',  # Tamaño del texto del brand
        'font-weight': 'bold',  # Negrilla
        'color': 'white',  # Color blanco para resaltar en barra oscura
    },
        style={
        'border': '2px solid #28a745',
        'border-radius': '10px',
        'font-size': '24px',  # Tamaño del texto
        'font-weight': 'bold'  # Negrilla opcional
    }
    ),
    html.Div(id='page-content')  # Aquí se renderizan las páginas
])

# Callback de la barra de navegacion
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname in [None, '/']:
        return home_layout  # Ruta predeterminada
    elif pathname == '/home':
        return home_layout
    elif pathname == '/data-exploration':
        return data_exploration_layout
    elif pathname == '/predictions':
        return predictions_layout
    else:
        return home_layout  # Si no coincide, regresa a Inicio

# Callback para actualizar el estado activo
@app.callback(
    [Output('home-link', 'active'),
     Output('data-exploration-link', 'active'),
     Output('predictions-link', 'active')],
    [Input('url', 'pathname')]
)
def update_active_links(pathname):
    # Compara la ruta actual con cada enlace y marca como activo si coincide
    return [
        pathname == "/home",
        pathname == "/data-exploration",
        pathname == "/predictions"
    ]

    
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)