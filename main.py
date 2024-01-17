import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import logging
from pandas import DataFrame
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import statsmodels.api as sm

def forward_selection(X, y, significance_level=0.05):
    initial_features = X.columns.tolist()
    best_features = []

    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[best_features + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features


def backward_elimination(X, y, significance_level = 0.05):
    features = X.columns.tolist()
    while len(features) > 0:
        features_with_constant = sm.add_constant(X[features])
        p_values = sm.OLS(y, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if max_p_value >= significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Load data
data = pd.read_csv('messy_data.csv')
logging.info("Data loaded successfully")
logging.info(data)

# Clean data
data.columns = data.columns.str.strip()
data = data.replace('', np.nan)
data = data.replace(' ', np.nan)

# Zmień wartości w kolumnach 'clarity', 'color' oraz 'cut' na lower-case
data['clarity'] = data['clarity'].str.strip()
data['clarity'] = data['clarity'].str.lower()

data['color'] = data['color'].str.strip()
data['color'] = data['color'].str.lower()

data['cut'] = data['cut'].str.strip()
data['cut'] = data['cut'].str.lower()

# Wyświetl unikalne wartości dla kolumn 'clarity', 'color', 'cut'
unique_clarity = data['clarity'].unique()
unique_color = data['color'].unique()
unique_cut = data['cut'].unique()

print("Unikalne wartości clarity:", unique_clarity)
print("Unikalne wartości color:", unique_color)
print("Unikalne wartości cut:", unique_cut)

# Nadaj etykiety liczbowe tym wartościom
label_encoder_clarity = LabelEncoder()
label_encoder_color = LabelEncoder()
label_encoder_cut = LabelEncoder()

num_data = DataFrame()
num_data['carat'] = data['carat']
num_data['x dimension'] = data['x dimension']
num_data['y dimension'] = data['y dimension']
num_data['z dimension'] = data['z dimension']
num_data['depth'] = data['depth']
num_data['table'] = data['table']
num_data['price'] = data['price']
num_data['clarity'] = label_encoder_clarity.fit_transform(data['clarity'])
num_data['color'] = label_encoder_color.fit_transform(data['color'])
num_data['cut'] = label_encoder_cut.fit_transform(data['cut'])

# Wyświetl mapowania
mapping_clarity = dict(zip(label_encoder_clarity.classes_, label_encoder_clarity.transform(label_encoder_clarity.classes_)))
mapping_color = dict(zip(label_encoder_color.classes_, label_encoder_color.transform(label_encoder_color.classes_)))
mapping_cut = dict(zip(label_encoder_cut.classes_, label_encoder_cut.transform(label_encoder_cut.classes_)))

print("Mapowanie clarity na liczby:", mapping_clarity)
print("Mapowanie color na liczby:", mapping_color)
print("Mapowanie cut na liczby:", mapping_cut)

# Wyświetl pierwsze wiersze danych po kodowaniu
print(data[['clarity', 'clarity', 'color', 'color', 'cut', 'cut']].head())
print("Zakodowane dane")
print(num_data)
missing_data_records = data[data.isnull().any(axis=1)]

print(missing_data_records)

#Uzupełnianie brakujących danych
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(num_data)
data_imputed = pd.DataFrame(data_imputed, columns=num_data.columns)

num_data = data_imputed

# usuwanie wartości odstajacych
Q1 = num_data.quantile(0.25)
Q3 = num_data.quantile(0.75)
IQR = Q3 - Q1
num_data = num_data[~((num_data < (Q1 - 1.5 * IQR)) | (num_data > (Q3 + 1.5 * IQR))).any(axis=1)]

logging.info("Data cleaned successfully")
print(num_data)

# Wybieranie cech (X) i zmiennej celu (y)
X = num_data.drop('price', axis=1)
y = num_data['price']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Budowanie modelu przy użyciu Forward Selection
selected_features_forward = forward_selection(X_train, y_train)
model_forward = LinearRegression().fit(X_train[selected_features_forward], y_train)

# Budowanie modelu przy użyciu Backward Elimination
selected_features_backward = backward_elimination(X_train, y_train)
model_backward = LinearRegression().fit(X_train[selected_features_backward], y_train)

app = Dash(__name__)
app.layout = html.Div([
    html.H1("Dashbord"),
    dcc.Dropdown(
        id='xaxis-column',
        options=[{'label': i, 'value': i} for i in num_data.columns if i != 'price'],
        value='carat'  # default value
    ),
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='histogram'),
    html.H2("Próbka Danych"),
    html.Div(id='data-table'),
    html.H1("Regresja Liniowa - Forward Selection"),
    dcc.Dropdown(
        id='forward-feature-dropdown',
        options=[{'label': i, 'value': i} for i in selected_features_forward],
        value=selected_features_forward[0]  # default value
    ),
    dcc.Graph(id='forward-regression-plot'),

    html.H1("Regresja Liniowa - Backward Elimination"),
    dcc.Dropdown(
        id='backward-feature-dropdown',
        options=[{'label': i, 'value': i} for i in selected_features_backward],
        value=selected_features_backward[0]  # default value
    ),
    dcc.Graph(id='backward-regression-plot'),
])

# Callback dla modelu Forward Selection
@app.callback(
    Output('forward-regression-plot', 'figure'),
    [Input('forward-feature-dropdown', 'value')]
)
def update_forward_regression_plot(selected_feature):
    # Utwórz DataFrame tylko z wybranymi cechami
    X_pred_forward = pd.DataFrame(X_test[selected_features_forward])

    # Wypełnij wszystkie cechy wartościami średnimi, z wyjątkiem wybranej cechy
    for feature in selected_features_forward:
        if feature != selected_feature:
            X_pred_forward[feature] = X_train[feature].mean()

    # Dokonaj predykcji
    y_pred_forward = model_forward.predict(X_pred_forward)

    # Utwórz wykres
    fig_forward = go.Figure()
    fig_forward.add_trace(go.Scatter(x=X_test[selected_feature], y=y_test, mode='markers', name='Aktualne'))
    fig_forward.add_trace(go.Scatter(x=X_test[selected_feature], y=y_pred_forward, mode='lines', name='Przewidywana'))
    fig_forward.update_layout(title=f'Price/{selected_feature}',
                              xaxis_title=selected_feature,
                              yaxis_title='Price')

    return fig_forward

# Callback dla modelu Backward Elimination
@app.callback(
    Output('backward-regression-plot', 'figure'),
    [Input('backward-feature-dropdown', 'value')]
)
def update_backward_regression_plot(selected_feature):
    # Utwórz DataFrame tylko z wybranymi cechami
    X_pred_backward = pd.DataFrame(X_test[selected_features_backward])

    # Wypełnij wszystkie cechy wartościami średnimi, z wyjątkiem wybranej cechy
    for feature in selected_features_backward:
        if feature != selected_feature:
            X_pred_backward[feature] = X_train[feature].mean()

    # Dokonaj predykcji
    y_pred_backward = model_backward.predict(X_pred_backward)

    # Utwórz wykres
    fig_backward = go.Figure()
    fig_backward.add_trace(go.Scatter(x=X_test[selected_feature], y=y_test, mode='markers', name='Aktualne'))
    fig_backward.add_trace(go.Scatter(x=X_test[selected_feature], y=y_pred_backward, mode='lines', name='Przewidywane'))
    fig_backward.update_layout(title=f'Price/{selected_feature}',
                               xaxis_title=selected_feature,
                               yaxis_title='Price')

    return fig_backward

# Callback for updating the scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('xaxis-column', 'value')]
)
def update_scatter(xaxis_column_name):
    logging.info(f"Updating for: {xaxis_column_name}")
    if xaxis_column_name in num_data.columns and not num_data.empty:
        fig = px.scatter(num_data, x=xaxis_column_name, y='price', trendline="ols", title=f"Price/{xaxis_column_name}")
        return fig
    else:
        logging.warning("Brak Danychy lub kolumna nie istnieje")
        return px.scatter(title="Brak Danych")


# Callback for updating the histogram
@app.callback(
    Output('histogram', 'figure'),
    [Input('xaxis-column', 'value')]
)
def update_histogram(xaxis_column_name):
    logging.info(f"Updating histogram: {xaxis_column_name}")
    if xaxis_column_name in num_data.columns and not num_data.empty:
        fig = px.histogram(num_data, x=xaxis_column_name, title=f"Rozkład Wartości({xaxis_column_name})")
        return fig
    else:
        logging.warning("Brak Danychy lub kolumna nie istnieje")
        return px.scatter(title="Brak Danych")


# Callback for updating the data table
@app.callback(
    Output('data-table', 'children'),
    [Input('xaxis-column', 'value')]
)
def update_table(xaxis_column_name):
    logging.info(f"Updating data table: {xaxis_column_name}")
    if not num_data.empty and xaxis_column_name in num_data.columns:
        sample_size = min(len(num_data), 10)
        sample_data = num_data[[xaxis_column_name, 'price']].sample(n=sample_size).to_dict('records')
        table_header = [html.Thead(html.Tr([html.Th(xaxis_column_name), html.Th("Price")]))]
        table_body = [
            html.Tbody([html.Tr([html.Td(item[xaxis_column_name]), html.Td(item['price'])]) for item in sample_data])]
        return table_header + table_body
    else:
        logging.warning("Brak Danychy lub kolumna nie istnieje")
        return [html.Thead(html.Tr([html.Th("Brak Danych")]))]


# Run the app
if __name__ == '__main__':
    logging.info("Running Dash app")
    app.run_server(debug=True)
