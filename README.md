🏠 House Price Prediction using Linear Models

This project builds a machine learning pipeline to predict house prices using linear regression–based models.
The goal is not just accuracy, but correct preprocessing, modeling discipline, and interpretability.

📌 Problem Statement

Predict house sale prices based on structural, temporal, and categorical features using linear regression, and analyze model performance, limitations, and improvements using regularization.

🧠 Approach Overview

The project follows a proper ML workflow:

Exploratory Data Analysis (EDA)

Feature engineering

Handling categorical variables

Feature scaling

Baseline Linear Regression

Ridge & Lasso Regularization

Polynomial feature expansion

Pipeline construction

Model serialization (pickle)

🛠️ Feature Engineering
Numerical Features

LotArea_log

TotalBsmtSF_log

YearBuilt

YearRemodAdd

(Log transformation applied to handle skewness.)

Categorical Features (One-Hot Encoded)

MSZoning

LotConfig

BldgType

Exterior1st

CondGroup

🔄 Preprocessing Pipeline

PolynomialFeatures (degree=2) for numerical variables

StandardScaler for numeric scaling

OneHotEncoder for categorical variables

ColumnTransformer to apply transformations correctly

🤖 Models Used
1️⃣ Linear Regression (Baseline)

Used to establish baseline performance

Diagnosed underfitting

2️⃣ Ridge Regression (Final Model)

Handles multicollinearity

Improves coefficient stability

Best regularization parameter found via cross-validation
