import  pandas as pd
import numpy as np
from ydata_profiling import ProfileReport 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('StudentScore.xls', delimiter=',')
target = "writing score"

# profile = ProfileReport(data, title="Student Score Report", explorative=True)
# profile.to_file("student_score_report.html")

x = data.drop([target], axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# imputer = SimpleImputer(strategy='median', missing_values= "NaN")
# x_train[["math score", "reading score"]] = imputer.fit_transform(x_train[["math score", "reading score"]])

# scaler = StandardScaler()
# x_train[["math score", "reading score"]] = scaler.fit_transform(x_train[["math score", "reading score"]])

num_transformers = Pipeline(steps=[
    ('imputer', SimpleImputer( missing_values= np.nan,strategy='median')),   
    ('scaler', StandardScaler())
])
result = num_transformers.fit_transform(x_train[["math score", "reading score"]])

education_values = ['some high school', 'high school', 'some college', 'associate\'s degree', 'bachelor\'s degree', 'master\'s degree']
gender_values = ['male', 'female']
lunch_values = data['lunch'].unique()
test_values = data['test preparation course'].unique()

ord_transformers = Pipeline(steps=[
    ('imputer', SimpleImputer( strategy='most_frequent')),   
    ('encoder', OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])

nom_transformers = Pipeline(steps=[
    ('imputer', SimpleImputer( strategy='most_frequent')),   
    ('encoder', OneHotEncoder(sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num_feature', num_transformers, ["math score", "reading score"]),
        ('ord_feature', ord_transformers, ["parental level of education", "gender", "lunch", "test preparation course"]),
        ('nom_feature', nom_transformers, ["race/ethnicity"]),
    ],
)

regression_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # ('model', LinearRegression())
    ('model', RandomForestRegressor())
])

# regression_model.fit(x_train, y_train)

# y_pred = regression_model.predict(x_test)

# for i, j in zip(y_test, y_pred):
#     print(f"Actual: {i}, Predicted: {j}")

# print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
# print("R2 Score - Coefficient of Determination:", r2_score(y_test, y_pred))
# print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Linear Regression 
# Mean Squared Error: 14.980822041816777
# R2 Score - Coefficient of Determination: 0.9378432907399291
# Mean Absolute Error: 3.2039447691582152

# Random Forest Regression
# Mean Squared Error: 19.85207288619331
# R2 Score - Coefficient of Determination: 0.9176320552268432
# Mean Absolute Error: 3.5789315476190473

# Vì sao mô hình phức tạp hơn lại có performance kém hơn?
# Vì có 2 feature là "math score" và "reading score" có mối quan hệ tuyến tính với "writing score",
# nên mô hình tuyến tính sẽ hoạt động tốt hơn.
# Mô hình Random Forest có thể hoạt động tốt hơn khi có nhiều feature không tuyến tính

# Hyperparameter tuning
# Sử dụng GridSearchCV 

params = {
    'model__n_estimators': [100, 200, 300],
    'model__criterion': ["squared_error", "absolute_error", "poisson"],
}

grid_search = GridSearchCV(
    estimator=regression_model,
    param_grid=params,
    cv=4,
    scoring='r2',
    verbose=2
)

grid_search.fit(x_train, y_train)

y_pred = grid_search.predict(x_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score - Coefficient of Determination:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))