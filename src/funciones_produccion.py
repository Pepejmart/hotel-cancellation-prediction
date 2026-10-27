import pandas as pd
import numpy as np
def get_ADR(df_reservas):
  # Para calcular el ADR dividimos el VALOR_RESERVA / NOCHES
  # Nota: Ya nos aseguramos que NOCHES > 0 en la limpieza, asi que no habra division por cero.

  df_reservas['ADR'] = df_reservas['VALOR_RESERVA'] / df_reservas['NOCHES']

  # 2 Verificación de seguridad
  print("--- Validación de la columna ADR (USD) ---")
  print(df_reservas[['NOCHES', 'VALOR_RESERVA', 'ADR']].head(10))

  # Estadisticas rapidas del nuevo ADR
  print("\nEstadisticas de ADR:")
  print(df_reservas['ADR'].describe())

  return df_reservas
def get_REV_PAX(df_reservas):
  # Para calcular el REV_PAX dividimos el VALOR_RESERVA / PAX
  # Nota: PAX >= 1 garantizado por la limpieza anterior si hubiéramos filtrado PAX=0 (lo asumimos >0 en reservas normales)
  df_reservas['REV_PAX'] = df_reservas['VALOR_RESERVA'] / df_reservas['PAX']
  return df_reservas
def get_LEAD_TIME(df_reservas):
  tamaño_antes = df_reservas.shape[0]

  # Normalización y Cálculo
  # Normalizar (Quitar horas) para forzar cálculo por calendario puro
  df_reservas['FECHA_TOMA'] = df_reservas['FECHA_TOMA']#.dt.normalize()
  df_reservas['LLEGADA'] = df_reservas['LLEGADA']#.dt.normalize() # Por seguridad

  # Calcular Lead Time
  df_reservas['LEAD_TIME'] = (df_reservas['LLEGADA'].dt.normalize() - df_reservas['FECHA_TOMA'].dt.normalize()).dt.days

  return df_reservas
def get_PAIS_TOP_200 (df_reservas):
  total_reservas_pais = df_reservas.groupby('PAIS')['ID_RESERVA'].count().reset_index()
  total_reservas_pais.columns = ['PAIS', 'Total_Reservas']

  # Top 200 paises
  top_200_paises = total_reservas_pais.sort_values(by='Total_Reservas', ascending=False).head(200)['PAIS'].tolist()

  # Nueva columna 'PAIS_TOP_200', con valor 'OTROS' para el resto
  df_reservas['PAIS_TOP_200'] = df_reservas['PAIS'].apply(lambda x: x if x in top_200_paises else 'OTROS')

  return df_reservas
def get_HORA_TOMA(df_reservas):
  df_reservas['HORA_TOMA'] = df_reservas['FECHA_TOMA'].dt.hour
  print('[OK] Columna HORA_TOMA creada correctamente.')
  display(df_reservas[['FECHA_TOMA', 'HORA_TOMA']].head())

  df_reservas['HORA_TOMA_SIN'] = np.sin(2 * np.pi * df_reservas['HORA_TOMA'] / 24)
  df_reservas['HORA_TOMA_COS'] = np.cos(2 * np.pi * df_reservas['HORA_TOMA'] / 24)

  return df_reservas
def get_TIPO_VIAJERO(df_reservas):
  # 1. Definimos las condiciones basadas en el número de personas (PAX)
  condiciones_pax = [
      (df_reservas['PAX'] == 1),
      (df_reservas['PAX'] == 2),
      (df_reservas['PAX'] >= 3)
  ]

  # 2. Definimos las etiquetas correspondientes
  etiquetas_pax = ['Single', 'Parejas', 'Familias']

  # 3. Creamos la nueva columna 'TIPO_VIAJERO'
  df_reservas['TIPO_VIAJERO'] = np.select(condiciones_pax, etiquetas_pax, default='Otro')

  # Porcentaje
  print("\nDistribución Porcentual:")
  print(df_reservas['TIPO_VIAJERO'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

  print("Distribución de Tipo de Viajero:")
  print("-" * 30)
  resultado = df_reservas['TIPO_VIAJERO'].value_counts()
  print(resultado)

  return df_reservas
def get_LOYALTY_REWARDS_y_TIENE_FIDELIDAD(df_reservas):

  # 1. Creamos la columna TIENE_FIDELIDAD
  # Si es NaN ponemos 0 si tiene cualquier nombre ponemos 1
  df_reservas['TIENE_FIDELIDAD'] = np.where(df_reservas['FIDELIDAD'].isna(), 0, 1)

  # 2. Creamos la columna LOYALTY_REWARDS
  # Si FIDELIDAD es Aurea Rewards ponemos 1 si no ponemos 0
  df_reservas['LOYALTY_REWARDS'] = np.where(df_reservas['FIDELIDAD'] == 'Aurea Rewards', 1, 0)

  return df_reservas
def get_ES_GRUPO(df_reservas):
  # 1. Creamos la columna ES_GRUPO
  # Si ID_MULTIPLE es distinto a 0 entonces ES_GRUPO es 1 si no ponemos 0
  df_reservas['ES_GRUPO'] = np.where(df_reservas['ID_MULTIPLE'] != 0, 1, 0)

  # 2. Validación de Negocio
  total_grupos = df_reservas['ES_GRUPO'].sum()
  print(f"--- ANÁLISIS DE SEGMENTACIÓN DE GRUPOS ---")
  print(f"📦 Reservas de Grupo: {total_grupos} ({(total_grupos/len(df_reservas))*100:.2f}%)")
  print(f"👤 Reservas Individuales: {len(df_reservas) - total_grupos}")

  # 3. Borramos el ID_MULTIPLE para limpiar el dataset (ya tenemos su valor en la nueva columna)
  if 'ID_MULTIPLE' in df_reservas.columns:
      df_reservas.drop(columns=['ID_MULTIPLE'], inplace=True)
      print("\n✅ Variable 'ES_GRUPO' creada e 'ID_MULTIPLE' eliminado.")
  else:
      print("\n⚠️ La columna 'ID_MULTIPLE' no existe (quizas ya se borro).")

  return df_reservas
def get_MES_LLEGADA(df_reservas):
  df_reservas['LLEGADA'] = pd.to_datetime(df_reservas['LLEGADA'])
  df_reservas['MES_LLEGADA'] = df_reservas['LLEGADA'].dt.month

  df_reservas['MES_LLEGADA_SIN'] = np.sin(2 * np.pi * df_reservas['MES_LLEGADA'] / 12)
  df_reservas['MES_LLEGADA_COS'] = np.cos(2 * np.pi * df_reservas['MES_LLEGADA'] / 12)

  return df_reservas
def get_NOMBRE_COMPLEJO(df_reservas):

  condiciones = [
      df_reservas['NOMBRE_HOTEL'] == 'Complejo Riviera Maya',
      df_reservas['NOMBRE_HOTEL'] == 'Complejo Punta Cana',
      df_reservas['NOMBRE_HOTEL'] == 'Complejo Costa Mujeres'
  ]

  opciones = ['MAYA', 'CANA', 'MUJE']

  df_reservas['NOMBRE_COMPLEJO'] = np.select(condiciones, opciones, default='OTRO')

  display(df_reservas[['NOMBRE_COMPLEJO', 'NOMBRE_HOTEL']].value_counts())

  return df_reservas
def get_PREFIJO_HAB(df_reservas):
  df_reservas['PREFIJO_HAB'] = [nombre.split(' ')[0] for nombre in df_reservas['NOMBRE_HABITACION']]
  filtro_cmu_fs = df_reservas['NOMBRE_HABITACION'].str.startswith('CMU FS')
  df_reservas.loc[filtro_cmu_fs, 'PREFIJO_HAB'] = 'CMU FS'
  df_reservas['PREFIJO_HAB'].unique()

  return df_reservas
def get_HOTEL_COMPLEJO(df_reservas):
  condiciones = [
      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Riviera Maya') & (df_reservas['PREFIJO_HAB'].str.startswith('COL')),
      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Riviera Maya') & (df_reservas['PREFIJO_HAB'].str.startswith('KAN')),
      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Riviera Maya') & (df_reservas['PREFIJO_HAB'].str.startswith('SIG')),
      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Riviera Maya') & (df_reservas['PREFIJO_HAB'].str.startswith('WS')),

      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Punta Cana') & (df_reservas['PREFIJO_HAB'].str.startswith('BAV')),
      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Punta Cana') & (df_reservas['PREFIJO_HAB'].str.startswith('PAL')),
      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Punta Cana') & (df_reservas['PREFIJO_HAB'].str.startswith('PC')),
      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Punta Cana') & (df_reservas['PREFIJO_HAB'].str.startswith('SIG')),

      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Costa Mujeres') & (df_reservas['PREFIJO_HAB'].str.startswith('CMU FS')),
      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Costa Mujeres') & (df_reservas['PREFIJO_HAB'].str.startswith('CMU')),
      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Costa Mujeres') & (df_reservas['PREFIJO_HAB'].str.startswith('SIGC')),
      (df_reservas['NOMBRE_HOTEL'] == 'Complejo Costa Mujeres') & (df_reservas['PREFIJO_HAB'].str.startswith('SIG'))
  ]

  opciones = [
      'MAYA_COL', 'MAYA_KAN', 'MAYA_SIG', 'MAYA_WS',
      'CANA_BAV', 'CANA_PAL', 'CANA_PC', 'CANA_SIG',
      'MUJE_CMU_FS', 'MUJE_CMU', 'MUJE_SIGC', 'MUJE_SIG'
  ]

  df_reservas['HOTEL_COMPLEJO'] = np.select(condiciones, opciones, default='OTRO')

  display(df_reservas[['NOMBRE_HOTEL','PREFIJO_HAB','HOTEL_COMPLEJO']].value_counts().sort_index())

  return df_reservas
def get_COMPLEJO_RESERVA(df_reservas):
  df_reservas['COMPLEJO_RESERVA'] = df_reservas['HOTEL_COMPLEJO'] + ' - ' + df_reservas['NOMBRE_HABITACION']
  display(df_reservas[['HOTEL_COMPLEJO', 'NOMBRE_HABITACION', 'COMPLEJO_RESERVA']].head())
  display(df_reservas['COMPLEJO_RESERVA'].nunique())

  return df_reservas
def get_HABITACION_TOP(df_reservas):
  ESCOGER_PONCENTAJE = 0.003
  # Calcular la frecuencia de cada NOMBRE_HABITACION
  counts = df_reservas['NOMBRE_HABITACION'].value_counts()

  # Obtener los nombres de las habitaciones que superan el % de las reservas
  percentage_threshold = 0.01  # umbral del 1%
  percentage_threshold = ESCOGER_PONCENTAJE
  top_habitaciones = counts[counts / len(df_reservas) > percentage_threshold].index.tolist()

  # Crear la nueva columna 'HABITACION_TOP'
  df_reservas['HABITACION_TOP'] = df_reservas['NOMBRE_HABITACION'].apply(lambda x: x if x in top_habitaciones else 'OTROS')

  # Verificar la distribución de la nueva columna
  display(df_reservas['HABITACION_TOP'].nunique())
  # display(df_reservas['HABITACION_TOP'].value_counts())
  # display(df_reservas['HABITACION_TOP'].value_counts(normalize=True))

  return df_reservas
def get_HOTEL_HABITACION_TOP (df_reservas):
  df_reservas['HOTEL_HABITACION_TOP'] = df_reservas['HOTEL_COMPLEJO'] + ' - ' + df_reservas['HABITACION_TOP']
  display(df_reservas['HOTEL_HABITACION_TOP'].nunique())
  # display(df_reservas['HOTEL_HABITACION_TOP'].value_counts())
  display(df_reservas['HOTEL_HABITACION_TOP'].value_counts(normalize=True))

  return df_reservas
def get_FUENTE_NEGOCIO_SEGMENTO_CLIENTE (df_reservas):
  display('Valores de FUENTE_NEGOCIO:', df_reservas['FUENTE_NEGOCIO'].value_counts().size)
  display('Valores de SEGMENTO:', df_reservas['SEGMENTO'].value_counts().size)
  display('Valores de CLIENTE:', df_reservas['CLIENTE'].value_counts().size)
  df_reservas['FUENTE_NEGOCIO_SEGMENTO_CLIENTE'] = df_reservas['FUENTE_NEGOCIO'] + '_' + df_reservas['SEGMENTO'] + '_' + df_reservas['CLIENTE']
  display('Valores de FUENTE_NEGOCIO_SEGMENTO_CLIENTE:', df_reservas['FUENTE_NEGOCIO_SEGMENTO_CLIENTE'].value_counts().size)
  display(df_reservas['FUENTE_NEGOCIO_SEGMENTO_CLIENTE'].head(5))

  return df_reservas
def get_features_df(df_reservas_in):

  df_reservas_in['LLEGADA'] = pd.to_datetime(df_reservas_in['LLEGADA'])
  df_reservas_in['FECHA_TOMA'] = pd.to_datetime(df_reservas_in['FECHA_TOMA'])

  df_reservas_in = get_ADR(df_reservas_in)
  df_reservas_in = get_REV_PAX(df_reservas_in)
  df_reservas_in = get_LEAD_TIME(df_reservas_in)
  df_reservas_in['PAIS_TOP_200'] = df_reservas_in['PAIS']
  df_reservas_in = get_HORA_TOMA(df_reservas_in)
  df_reservas_in = get_TIPO_VIAJERO(df_reservas_in)
  df_reservas_in = get_LOYALTY_REWARDS_y_TIENE_FIDELIDAD(df_reservas_in)
  df_reservas_in = get_ES_GRUPO(df_reservas_in)
  df_reservas_in = get_MES_LLEGADA(df_reservas_in)
  df_reservas_in = get_NOMBRE_COMPLEJO(df_reservas_in)
  df_reservas_in = get_PREFIJO_HAB(df_reservas_in)
  df_reservas_in = get_HOTEL_COMPLEJO(df_reservas_in)
  df_reservas_in = get_COMPLEJO_RESERVA(df_reservas_in)
  df_reservas_in = get_HABITACION_TOP(df_reservas_in)
  df_reservas_in = get_HOTEL_HABITACION_TOP (df_reservas_in)
  df_reservas_in = get_FUENTE_NEGOCIO_SEGMENTO_CLIENTE (df_reservas_in)

  return df_reservas_in
def get_features(LLEGADA, NOCHES, PAX, ADULTOS, CLIENTE, FECHA_TOMA, FIDELIDAD, PAIS, SEGMENTO, FUENTE_NEGOCIO, NOMBRE_HOTEL, NOMBRE_HABITACION, VALOR_RESERVA):
  data = {
      'LLEGADA': [LLEGADA],
      'NOCHES': [NOCHES],
      'PAX': [PAX],
      'ADULTOS': [ADULTOS],
      'CLIENTE': [CLIENTE],
      'ID_MULTIPLE': 0,
      'FECHA_TOMA': [FECHA_TOMA],
      'FIDELIDAD': [FIDELIDAD],
      'PAIS': [PAIS],
      'SEGMENTO': [SEGMENTO],
      'FUENTE_NEGOCIO': [FUENTE_NEGOCIO],
      'NOMBRE_HOTEL': [NOMBRE_HOTEL],
      'NOMBRE_HABITACION': [NOMBRE_HABITACION],
      'VALOR_RESERVA': [VALOR_RESERVA]
  }

  df_reservas_in = pd.DataFrame(data)

  df_reservas_in['LLEGADA'] = pd.to_datetime(df_reservas_in['LLEGADA'])
  df_reservas_in['FECHA_TOMA'] = pd.to_datetime(df_reservas_in['FECHA_TOMA'])

  df_reservas_in = get_ADR(df_reservas_in)
  df_reservas_in = get_REV_PAX(df_reservas_in)
  df_reservas_in = get_LEAD_TIME(df_reservas_in)
  df_reservas_in['PAIS_TOP_200'] = df_reservas_in['PAIS']
  df_reservas_in = get_HORA_TOMA(df_reservas_in)
  df_reservas_in = get_TIPO_VIAJERO(df_reservas_in)
  df_reservas_in = get_LOYALTY_REWARDS_y_TIENE_FIDELIDAD(df_reservas_in)
  df_reservas_in = get_ES_GRUPO(df_reservas_in)
  df_reservas_in = get_MES_LLEGADA(df_reservas_in)
  df_reservas_in = get_NOMBRE_COMPLEJO(df_reservas_in)
  df_reservas_in = get_PREFIJO_HAB(df_reservas_in)
  df_reservas_in = get_HOTEL_COMPLEJO(df_reservas_in)
  df_reservas_in = get_COMPLEJO_RESERVA(df_reservas_in)
  df_reservas_in = get_HABITACION_TOP(df_reservas_in)
  df_reservas_in = get_HOTEL_HABITACION_TOP (df_reservas_in)
  df_reservas_in = get_FUENTE_NEGOCIO_SEGMENTO_CLIENTE (df_reservas_in)

  return df_reservas_in
