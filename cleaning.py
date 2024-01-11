import pandas as pd


def read_csv_data(path: str) -> pd.DataFrame:
    """
    A function to read csv data using pandas for the cleaning process.
    
    Args:
        path (str): The path to the file to read.
    
    Returns:
        df (pd.DataFrame): The df with the info to clean
    """
    try:
        raw_data = pd.read_csv(path, sep=',')
        
        return raw_data
    
    except FileNotFoundError as e:
        print(f'Error: {e}. The file {path} does not exist.')
    except pd.errors.EmptyDataError as e:
        print(f'Error: {e}. The file {path} is empty')
    except Exception as e:
        print(f'Error: {e}. An unexpected error ocurred.')
        
def make_binary_cols(df: pd.DataFrame, dict_to_replace: dict) -> pd.DataFrame:
    """
    Reemplaza los datos categoricos de tipo binario por 0 y 1, para todas las columnas especificada
    en el diccionario.
    
    Args:
        df (pd.DataFrame): El dataframe en el que reemplazaremos los valores por valores binarios.
        dict_to_replace (dict): Diccionario con los valores para reemplazar.
        
    Returns:
        df_clean (pd.DataFrame): Dataframe con los valores reemplazados.
    """
    df_clean = df.replace(dict_to_replace)
    return df_clean