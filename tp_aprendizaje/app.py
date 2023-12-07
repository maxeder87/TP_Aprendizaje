import streamlit as st
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import os

# Cargar el modelo
path_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(path_dir, 'modelo.pkl')
loaded_model = joblib.load(pkl_path)

data_path = os.path.join(path_dir, 'train_data.csv')
train_data = pd.read_csv(data_path)

# Separamos los datos (son todos de entrenamiento)
X = train_data.drop('target', axis=1)
y = train_data['target']

# Creamos un pipeline con imputador, normalización y el modelo
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', loaded_model)
])

# Entrenamos el pipeline 
pipeline.fit(X, y)

st.title('Predicción de lluvia en Australia')

def get_user_input():
    # Proporcionamos un formulario para que los usuarios ingresen valores
    input_dict = {}
    for variable in ['MinTemp', 'MaxTemp', 'Rainfall', 'Sunshine', 'WindGustSpeed',
                     'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                     'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'RainToday', 'WindGustDir']:
        if variable == 'RainToday':
            # Permitimos a los usuarios seleccionar "Sí" o "No" y asignamos 1 o 0 respectivamente
            input_value = 1 if st.selectbox(f"Seleccionar {variable}", ['No', 'Yes']) == 'Yes' else 0
        elif variable == 'WindGustDir':
            # Para 'WindGustDir', permitimos a los usuarios seleccionar de una lista
            wind_direction = st.radio(f"Seleccionar dirección del viento para {variable}", ['E', 'N', 'S', 'W'])
            # Asignamos 1 a la dirección seleccionada y 0 a las demás
            input_dict['WindGustDir_E'] = 1 if wind_direction == 'E' else 0
            input_dict['WindGustDir_N'] = 1 if wind_direction == 'N' else 0
            input_dict['WindGustDir_S'] = 1 if wind_direction == 'S' else 0
            input_dict['WindGustDir_W'] = 1 if wind_direction == 'W' else 0
        else:
            # Utilizamos sliders para las variables numéricas
            min_value = train_data[variable].min()
            max_value = train_data[variable].max()
            input_value = st.slider(f"Seleccionar valor para {variable}", min_value=min_value, max_value=max_value, value=min_value, step=0.1)
            input_dict[variable] = input_value
    submit_button = st.button('Submit')
    return input_dict, submit_button

user_input, submit_button = get_user_input()

if submit_button:
    # Creamos un DataFrame con los datos del usuario
    user_data = pd.DataFrame([user_input])

    # Aseguramos que las columnas en user_data coincidan con las columnas en X
    user_data = user_data.reindex(columns=X.columns, fill_value=0)

    # Transformamos los datos del usuario usando el mismo preprocesamiento que en X
    transformed_data = pipeline.named_steps['imputer'].transform(user_data)
    transformed_data = pipeline.named_steps['scaler'].transform(transformed_data)

    # Realizamos la predicción con el modelo
    prediction = pipeline.named_steps['model'].predict(transformed_data)
    prediction_value = prediction[0]

    # Mostramos la predicción
    st.header("¿Lloverá mañana?")
    prediction_label = 'Lloverá' if prediction_value == 1 else 'No lloverá'
    st.write(prediction_label)