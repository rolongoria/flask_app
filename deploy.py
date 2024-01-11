import joblib

def save_model(model):
    
    joblib.dump(model, 'models/modelo_prueba_1.joblib')
    
    return print('Modelo guardado con exito')