import pandas as pd
import numpy as np
import os

def limpiar_dataset():
    # Detectar el directorio actual para encontrar el archivo
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    archivo_entrada = os.path.join(directorio_actual, 'conapesca_historico_2015_2025.csv')
    archivo_salida = os.path.join(directorio_actual, 'conapesca_historico_2015_2025_LIMPIO.csv')
    
    print(f"Leyendo archivo: {archivo_entrada}...")
    try:
        df = pd.read_csv(archivo_entrada, low_memory=False)
    except FileNotFoundError:
        print("Error: No se encontró el archivo. Asegúrate de que este script esté en la misma carpeta que el CSV.")
        return
        
    print(f"Registros originales: {df.shape[0]} filas y {df.shape[1]} columnas.")
    
    print("1. Unificando columnas desfasadas por cambios de nombre entre años...")
    # Unificar AÑO / ANO
    if 'ANO' in df.columns and 'AÑO' in df.columns:
        df['ANIO_OFICIAL'] = df['ANO'].combine_first(df['AÑO'])
        df.drop(columns=['ANO', 'AÑO'], inplace=True)
        
    # Unificar NOMBREPRINCIPALESPECIE / NOMBRE PRINCIPAL ESPECIE
    if 'NOMBREPRINCIPALESPECIE' in df.columns and 'NOMBRE PRINCIPAL ESPECIE' in df.columns:
        df['ESPECIE'] = df['NOMBREPRINCIPALESPECIE'].combine_first(df['NOMBRE PRINCIPAL ESPECIE'])
        df.drop(columns=['NOMBREPRINCIPALESPECIE', 'NOMBRE PRINCIPAL ESPECIE'], inplace=True)
        
    # Unificar ORIGEN DEL CAMARON / ORIGEN DEL CAMARÓN
    if 'ORIGEN DEL CAMARON' in df.columns and 'ORIGEN DEL CAMARÓN' in df.columns:
        df['ORIGEN_CAMARON'] = df['ORIGEN DEL CAMARON'].combine_first(df['ORIGEN DEL CAMARÓN'])
        df.drop(columns=['ORIGEN DEL CAMARON', 'ORIGEN DEL CAMARÓN'], inplace=True)
        
    # Unificar TIPOGUA / TIPO AGUA
    if 'TIPOGUA' in df.columns and 'TIPO AGUA' in df.columns:
        df['TIPO_AGUA'] = df['TIPOGUA'].combine_first(df['TIPO AGUA'])
        df.drop(columns=['TIPOGUA', 'TIPO AGUA'], inplace=True)

    print("2. Estandarizando nombres de columnas (mayúsculas, sin espacios)...")
    # Quitar paréntesis, acentos y cambiar espacios por guiones bajos
    df.columns = (df.columns.str.strip()
                  .str.upper()
                  .str.replace(' ', '_', regex=False)
                  .str.replace('(', '', regex=False)
                  .str.replace(')', '', regex=False)
                  .str.replace('Ó', 'O', regex=False))

    print("3. Limpiando texto (quitando espacios extra y estandarizando mayúsculas)...")
    cols_categoricas = df.select_dtypes(include=['object']).columns
    for col in cols_categoricas:
        df[col] = df[col].astype(str).str.strip().str.upper()
        # Eliminar acentos para evitar duplicados como "PACÍFICO" y "PACIFICO"
        df[col] = df[col].str.replace('Á', 'A').str.replace('É', 'E').str.replace('Í', 'I').str.replace('Ó', 'O').str.replace('Ú', 'U')
        df[col] = df[col].replace(['NAN', 'NONE', 'NULL'], np.nan)
        
    print("4. Manejando valores nulos (NaN)...")
    # Texto nulo a "NO ESPECIFICADO"
    df[cols_categoricas] = df[cols_categoricas].fillna('NO ESPECIFICADO')
    # Números nulos a 0
    cols_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    df[cols_numericas] = df[cols_numericas].fillna(0)
    
    # Convertir el año a entero
    if 'ANIO_OFICIAL' in df.columns:
        df['ANIO_OFICIAL'] = df['ANIO_OFICIAL'].astype(int)

    print("5. Guardando archivo limpio...")
    df.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
    print(f"¡Limpieza terminada! Dataset guardado en: {archivo_salida}")
    print(f"Dataset final: {df.shape[0]} filas y {df.shape[1]} columnas limpias.")

if __name__ == '__main__':
    limpiar_dataset()
