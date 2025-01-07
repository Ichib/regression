'''Building Streamlit Application'''
# build a streamlit web application to predict the petal width of an iris flower based on the petal length, sepal length, and species of the flower
# The web application will use a multiple linear regression model implemented using PyTorch to make predictions
# The web application will be built using the Streamlit library
# The web application will be deployed to Heroku
# Use model in src/P1/PDS-Regression/TP-Regression.ipynb on Part 3 Multiple Linear Regression to use in the web application
# import the required libraries
import torch
import torch.nn as nn
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')


# Multiple Linear Regression Model (For Petal Width Prediction)
class MultipleLinearRegression(nn.Module):
    """
    Multiple linear regression model for petal width prediction
    """
    def __init__(self, input_dim):
        super(MultipleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# Logistic Regression Model (For Species Classification)
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(4, 1)  # 4 input features, 1 output (binary)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Multi-Class Logistic Regression Model (For Penguins Classification)
class MultiClassLogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiClassLogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# MPG Logistic Regression Model (For MPG Classification)
class MPGLogisticRegressionModel(nn.Module):
    def __init__(self):
        super(MPGLogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(4, 1)  # input features, 1 output (binary)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Load trained models and scalers
# Petal Width Prediction (Multiple Linear Regression)
input_dim = 3  # 3 features (petal length, sepal length, species)
pytorch_model_mlr = MultipleLinearRegression(input_dim)
pytorch_model_mlr.load_state_dict(
    torch.load('/model/pytorch_iris_mlr_model.pkl')
)
pytorch_model_mlr.eval()
sklearn_model_mlr = joblib.load('/Users/mac/Desktop/Home/Year 5/NLP/NLP-Project/src/P1/PDS-Regression/model/sklearn_iris_mlr_model.pkl')
scaler_mlr = joblib.load('/model/scaler_mlr_model.pkl')

# Species Classification (Binary Logistic Regression)
pytorch_model_binary = LogisticRegressionModel()
pytorch_model_binary.load_state_dict(
    torch.load('/model/pytorch_iris_binary_lgr_model.pkl')
)
pytorch_model_binary.eval()
sklearn_model_binary = joblib.load('/model/sklearn_iris_binary_lgr_model.pkl')
scaler_binary = joblib.load('/model/scaler_binary_lgr_model.pkl')

# Penguins Classification (Multi-Class Logistic Regression)
input_dim_penguins = 6  # Number of input features for penguins classification
output_dim_penguins = 3  # Number of output classes for penguins classification
pytorch_model_penguins = MultiClassLogisticRegressionModel(input_dim_penguins, output_dim_penguins)
pytorch_model_penguins.load_state_dict(
    torch.load('/model/pytorch_penguins_model.pkl')
)
pytorch_model_penguins.eval()
sklearn_model_penguins = joblib.load('/model/sklearn_penguins_model.pkl')
scaler_penguins = joblib.load('/model/scaler_penguins_model.pkl')

# MPG Classification (Binary Logistic Regression)
pytorch_model_mpg = MPGLogisticRegressionModel()
pytorch_model_mpg.load_state_dict(
    torch.load('/model/pytorch_mpg_model.pkl')
)
pytorch_model_mpg.eval()
sklearn_model_mpg = joblib.load('/model/sklearn_mpg_model.pkl')
scaler_mpg = joblib.load('/model/scaler_mpg_model.pkl')


# Streamlit UI
st.title('🌸🐧 Model Prediction')

# Choose between Petal Width Prediction and Species Classification
problem_choice = st.selectbox('Choose the problem to work on', options=['🌸 Petal Width Prediction', '🌸 Species Classification', '🐧 Penguins Classification', '🚗 MPG Binary Logistic Regression'])

if problem_choice == '🌸 Petal Width Prediction':
    st.write('Enter the details of the iris flower to predict its petal width.')

    # Input fields for petal width prediction (MLR model)
    petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=1.5)
    sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=4.6)
    species = st.selectbox('Species', options=[0, 1, 2], format_func=lambda x: ['Setosa', 'Versicolor', 'Virginica'][x])

    model_choice = st.selectbox('Choose Model', options=['PyTorch', 'Sklearn'])

    if st.button('Predict'):
        input_data = [[petal_length, sepal_length, species]]
        input_data_scaled = scaler_mlr.transform(input_data)
        
        if model_choice == 'PyTorch':
            input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
            with torch.no_grad():
                prediction = pytorch_model_mlr(input_tensor).item()
        else:
            prediction = sklearn_model_mlr.predict(input_data_scaled)[0]
        
        st.write(f'🌸 The predicted petal width is {prediction:.2f}')

elif problem_choice == '🌸 Species Classification':
    st.write('Enter the details of the iris flower to classify its species.')

    # Input fields for species classification (Binary Logistic Regression model)
    petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=1.5)
    sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=4.6)
    petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, value=0.5)
    sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, value=3.0)

    model_choice = st.selectbox('Choose Model', options=['PyTorch', 'Sklearn'])

    if st.button('Predict'):
        # Now using 4 features: petal_length, sepal_length, petal_width, sepal_width
        input_data = [[petal_length, sepal_length, petal_width, sepal_width]]  # 4 features
        input_data_scaled = scaler_binary.transform(input_data)
        
        if model_choice == 'PyTorch':
            input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
            with torch.no_grad():
                prediction = pytorch_model_binary(input_tensor).item()
                prediction = 1 if prediction >= 0.5 else 0  # Convert output to binary (Setosa or Not Setosa)
        else:
            prediction = sklearn_model_binary.predict(input_data_scaled)[0]
        
        species = ['Not Setosa', 'Setosa']
        st.write(f'🌸 The flower is predicted to be: {species[prediction]}')

elif problem_choice == '🐧 Penguins Classification':
    st.write('Enter the details of the penguin to classify its species.')

    # Input fields for penguins classification (Multi-Class Logistic Regression model)
    island = st.selectbox('Island', options=['Biscoe', 'Dream', 'Torgersen'])
    bill_length_mm = st.number_input('Bill Length (mm)', min_value=0.0, max_value=100.0, value=45.0)
    bill_depth_mm = st.number_input('Bill Depth (mm)', min_value=0.0, max_value=30.0, value=15.0)
    flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0, max_value=300.0, value=200.0)
    body_mass_g = st.number_input('Body Mass (g)', min_value=0.0, max_value=10000.0, value=4000.0)
    sex = st.selectbox('Sex', options=['Male', 'Female'])

    model_choice = st.selectbox('Choose Model', options=['PyTorch', 'Sklearn'])

    if st.button('Predict'):
        # Encode categorical features
        island_mapping = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
        sex_mapping = {'Male': 0, 'Female': 1}
        island_encoded = island_mapping[island]
        sex_encoded = sex_mapping[sex]

        # Now using 6 features: island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex
        input_data = [[island_encoded, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_encoded]]  # 6 features
        input_data_scaled = scaler_penguins.transform(input_data)
        
        if model_choice == 'PyTorch':
            input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
            with torch.no_grad():
                prediction = pytorch_model_penguins(input_tensor).argmax(dim=1).item()
        else:
            prediction = sklearn_model_penguins.predict(input_data_scaled)[0]
        
        species = ['Adelie', 'Chinstrap', 'Gentoo']
        if species[prediction] == 'Adelie':
            st.write(f'🐧 The penguin is predicted to be: {species[prediction]}')
            st.image('/img/adelie_pen.jpeg', width=500)
        elif species[prediction] == 'Chinstrap':
            st.write(f'🐧 The penguin is predicted to be: {species[prediction]}')
            st.image('/img/chinstrap_pen.jpeg', width=500)
        else:
            st.write(f'🐧 The penguin is predicted to be: {species[prediction]}')
            st.image('/img/gentoo_pen.jpeg', width=500)

elif problem_choice == '🚗 MPG Binary Logistic Regression':
    st.write('Enter the details of the car to classify its MPG.')

    # Input fields for MPG classification (Binary Logistic Regression model)
    cylinders = st.number_input('Cylinders', min_value=1, max_value=12, value=4)
    displacement = st.number_input('Displacement', min_value=0.0, max_value=1000.0, value=150.0)
    horsepower = st.number_input('Horsepower', min_value=0.0, max_value=1000.0, value=100.0)
    weight = st.number_input('Weight', min_value=0.0, max_value=10000.0, value=3000.0)

    model_choice = st.selectbox('Choose Model', options=['PyTorch', 'Sklearn'])

    if st.button('Predict'):
        # Now using 4 features: cylinders, displacement, horsepower, weight
        input_data = [[cylinders, displacement, horsepower, weight]]  # 4 features
        input_data_scaled = scaler_mpg.transform(input_data)
        
        if model_choice == 'PyTorch':
            input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
            with torch.no_grad():
                prediction = pytorch_model_mpg(input_tensor).item()
                prediction = 1 if prediction >= 0.5 else 0  # Convert output to binary (High MPG or Low MPG)
        else:
            prediction = sklearn_model_mpg.predict(input_data_scaled)[0]
        
        mpg_class = ['Low MPG', 'High MPG']
        st.write(f'🚗 The car is predicted to have: {mpg_class[prediction]}')
