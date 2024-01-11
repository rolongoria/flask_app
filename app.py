from cleaning import read_csv_data, make_binary_cols
from training import train_random_forest, evaluate_model, make_train_test
from deploy import save_model

diabetes_data = read_csv_data("data/diabetes_risk_prediction_dataset.csv")

dict_to_replace = {
    'Yes' : 1, 'No' : 0,
    'Male' : 1, 'Female' : 0,
    'Positive' : 1, 'Negative' : 0}

clean_data = make_binary_cols(diabetes_data, dict_to_replace)

X_train, X_test, y_train, y_test = make_train_test(clean_data, 'class')

random_forest_model = train_random_forest(X_train, y_train)

score, report = evaluate_model(random_forest_model, X_test, y_test)

print(f'El modelo tiene un accuracy de: {score}')
print(report)

save_model(random_forest_model)