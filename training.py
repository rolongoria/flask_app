from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def make_train_test(train_data: pd.DataFrame, objetive_col: str) -> pd.DataFrame:
    X = train_data.drop(objetive_col, axis=1)
    y = train_data[objetive_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame):
    model = RandomForestClassifier(random_state=11)
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    model_accuracy = accuracy_score(y_test, y_pred)
    
    model_report = classification_report(y_test, y_pred)
    
    return model_accuracy, model_report
    