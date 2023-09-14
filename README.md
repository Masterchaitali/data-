# data-
data science internship project 
As part of the screening test, you will write code to parse the JSON file provided(algoparams_from_ui) and kick off in sequence the following machine learning steps programmatically. Keep in mind your final code should be able to parse any Json that follows this format. It is crucial you have a generic parse that can read the various steps like feature handling, feature generation and model building using Grid search after parsing hyper params.
1) Read the target and type of regression to be run
2) import json

# Your JSON data (replace this with your actual JSON data)
json_data = '''
{
    "session_name": "test",
    "session_description": "test",
    "design_state_data": {
        ...
        "target": {
            "prediction_type": "Regression",
            "target": "petal_width",
            "type": "regression",
            "partitioning": true
        },
        ...
    }
}
'''

# Parse the JSON data
data = json.loads(json_data)

# Extract the target variable and type of regression
target_variable = data["design_state_data"]["target"]["target"]
regression_type = data["design_state_data"]["target"]["type"]

# Print the extracted information
print("Target Variable:", target_variable)
print("Regression Type:", regression_type)

2) Read the features (which are column names in the csv) and figure out what missing imputation needs to be applied and apply that to the columns loaded in a dataframe
3) import json
import pandas as pd

# Your JSON data (replace this with your actual JSON data)
json_data = '''
{
    "session_name": "test",
    "session_description": "test",
    "design_state_data": {
        ...
        "feature_handling": {
            "sepal_length": {
                "feature_name": "sepal_length",
                "is_selected": true,
                "feature_variable_type": "numerical",
                "feature_details": {
                    "numerical_handling": "Keep as regular numerical feature",
                    "rescaling": "No rescaling",
                    "make_derived_feats": false,
                    "missing_values": "Impute",
                    "impute_with": "Average of values",
                    "impute_value": 0
                }
            },
            "sepal_width": {
                "feature_name": "sepal_width",
                "is_selected": true,
                "feature_variable_type": "numerical",
                "feature_details": {
                    "numerical_handling": "Keep as regular numerical feature",
                    "rescaling": "No rescaling",
                    "make_derived_feats": false,
                    "missing_values": "Impute",
                    "impute_with": "custom",
                    "impute_value": -1
                }
            },
            ...
        },
        ...
    }
}
'''

# Parse the JSON data
data = json.loads(json_data)

# Load your CSV data into a DataFrame (replace 'your_data.csv' with your actual CSV file path)
df = pd.read_csv('iris.csv')

# Extract feature handling information
feature_handling = data["design_state_data"]["feature_handling"]

# Iterate through the features and apply missing value imputation
for feature_name, details in feature_handling.items():
    if details["feature_details"]["missing_values"] == "Impute":
        impute_method = details["feature_details"]["impute_with"]
        impute_value = details["feature_details"]["impute_value"]
        
        if impute_method == "Average of values":
            df[feature_name].fillna(df[feature_name].mean(), inplace=True)
        elif impute_method == "custom":
            df[feature_name].fillna(impute_value, inplace=True)

# Now, your DataFrame 'df' contains the missing value imputations as specified in the JSON configuration.

2) Compute feature reduction based on input. See the screenshot below where there can be No Reduction, Corr with Target, Tree-based, PCA. Please make sure you write code so that all options can work. If we rerun your code with a different Json it should work if we switch No Reduction to say PCA.
import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder

# Your JSON data (replace this with your actual JSON data)
json_data = '''
{
    "session_name": "test",
    "session_description": "test",
    "design_state_data": {
        ...
        "feature_reduction": {
            "feature_reduction_method": "PCA"
        },
        ...
    }
}
'''

# Parse the JSON data
data = json.loads(json_data)

# Load your CSV data into a DataFrame (replace 'your_data.csv' with your actual CSV file path)
df = pd.read_csv('iris.csv')

# Extract the feature reduction method from the JSON
feature_reduction_method = data["design_state_data"]["feature_reduction"]["feature_reduction_method"]

# Define a label encoder for text features if needed
label_encoder = LabelEncoder()

# Apply the selected feature reduction method
if feature_reduction_method == "No Reduction":
    pass  # No reduction needed

elif feature_reduction_method == "Corr with Target":
    # Compute correlation between features and the target variable
    correlations = df.corr()
    target_correlations = correlations["target"].abs().sort_values(ascending=False)
    
    # Select the top N features based on correlation
    top_n_features = 5  # You can adjust this number
    selected_features = target_correlations[1:top_n_features + 1].index.tolist()
    df = df[selected_features]

elif feature_reduction_method == "Tree-based":
    # Separate features and target variable
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Create a tree-based model for feature selection
    if data["design_state_data"]["target"]["prediction_type"] == "Regression":
        model = RandomForestRegressor()
    else:
        model = RandomForestClassifier()
    
    # Fit the model to the data to compute feature importances
    model.fit(X, y)
    
    # Select features based on their importances
    feature_selector = SelectFromModel(model)
    feature_selector.fit(X, y)
    selected_features = X.columns[feature_selector.get_support()].tolist()
    df = df[selected_features]

elif feature_reduction_method == "PCA":
    # Separate features and target variable
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)  # You can adjust the number of components
    X_reduced = pca.fit_transform(X)
    
    # Create a new DataFrame with reduced features and add the target column
    df_reduced = pd.DataFrame(data=X_reduced, columns=["PCA1", "PCA2"])
    df_reduced["target"] = y
    df = df_reduced

# Now, 'df' contains the data with the selected feature reduction method applied.

5) Run the fit and predict on each model – keep in mind that you need to do hyper parameter tuning i.e., use GridSearchCV
6) import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score  # Replace with appropriate metrics for your problem
from sklearn.preprocessing import LabelEncoder

# Your JSON data (replace this with your actual JSON data)
json_data = '''
{
    "session_name": "test",
    "session_description": "test",
    "design_state_data": {
        ...
        "hyperparameters": {
            "stratergy": "Grid Search",
            ...
        },
        "algorithms": {
            "RandomForestRegressor": {
                "model_name": "Random Forest Regressor",
                "is_selected": true,
                ...
            },
            "LinearRegression": {
                "model_name": "LinearRegression",
                "is_selected": false,
                ...
            },
            ...
        },
        ...
    }
}
'''

# Parse the JSON data
data = json.loads(json_data)

# Load your CSV data into a DataFrame (replace 'your_data.csv' with your actual CSV file path)
df = pd.read_csv('iris.csv')

# Extract the target variable and features
X = df.drop(columns=["target"])
y = df["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a label encoder for text features if needed
label_encoder = LabelEncoder()

# Initialize a dictionary to store model results
model_results = {}

# Extract the list of selected algorithms from the JSON configuration
selected_algorithms = [model_name for model_name, model_data in data["design_state_data"]["algorithms"].items() if model_data["is_selected"]]

# Iterate through selected algorithms and perform hyperparameter tuning and prediction
for algorithm_name in selected_algorithms:
    algorithm_data = data["design_state_data"]["algorithms"][algorithm_name]
    
    # Define the model based on the algorithm name (you can add more models as needed)
    if algorithm_name == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
    elif algorithm_name == "LinearRegression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    # Add more model definitions as needed
    
    # Define hyperparameter grid for GridSearchCV (use values from your JSON)
    hyperparameter_grid = {
        # Specify hyperparameters and their possible values for the current model
        # Example:
        # "parameter_name": [value1, value2, ...]
        # Replace with actual hyperparameters and values from your JSON
    }
    
    # Initialize GridSearchCV with the model and hyperparameter grid
    grid_search = GridSearchCV(model, param_grid=hyperparameter_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # Fit the model with hyperparameter tuning on the training data
    grid_search.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = grid_search.predict(X_test)
    
    # Calculate evaluation metrics (replace with appropriate metrics for your problem)
    if algorithm_data["model_name"] == "Random Forest Regressor":
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_results[algorithm_name] = {"MSE": mse, "R2 Score": r2}
    elif algorithm_data["model_name"] == "LinearRegression":
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_results[algorithm_name] = {"MSE": mse, "R2 Score": r2}
    # Add more metric calculations as needed
    
# Print the results for each model
for model_name, results in model_results.items():
    print(f"Model: {model_name}")
    for metric, value in results.items():
        print(f"{metric}: {value}")
    print("\n")

# You can access the best hyperparameters and trained models from grid_search.best_params_ and grid_search.best_estimator_ respectively.

4) Parse the Json and make the model objects (using sklean) that can handle what is required in the “prediction_type” specified in the JSON (See #1 where “prediction_type” is specified). Keep in mind not to pick models that don’t apply for the prediction_type specified

5) import json
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
# Import other relevant scikit-learn models as needed

# Your JSON data (replace this with your actual JSON data)
json_data = '''
{
    "session_name": "test",
    "session_description": "test",
    "design_state_data": {
        ...
        "target": {
            "prediction_type": "Classification",  # Replace with "Regression" for regression tasks
            "target": "petal_width",
            "type": "classification",  # Replace with "regression" for regression tasks
            "partitioning": true
        },
        ...
        "algorithms": {
            "RandomForestClassifier": {
                "model_name": "Random Forest Classifier",
                "is_selected": true,
                ...
            },
            "RandomForestRegressor": {
                "model_name": "Random Forest Regressor",
                "is_selected": true,
                ...
            },
            "LogisticRegression": {
                "model_name": "LogisticRegression",
                "is_selected": false,
                ...
            },
            "LinearRegression": {
                "model_name": "LinearRegression",
                "is_selected": false,
                ...
            },
            ...
        },
        ...
    }
}
'''

# Parse the JSON data
data = json.loads(json_data)

# Extract the prediction type from the JSON configuration
prediction_type = data["design_state_data"]["target"]["prediction_type"].lower()  # Convert to lowercase for comparison

# Initialize a dictionary to store model objects
model_objects = {}

# Define models based on prediction type (classification or regression)
if prediction_type == "classification":
    # Create model objects for classification tasks
    for algorithm_name, algorithm_data in data["design_state_data"]["algorithms"].items():
        if algorithm_data["is_selected"]:
            if algorithm_name == "RandomForestClassifier":
                model = RandomForestClassifier()
            elif algorithm_name == "LogisticRegression":
                model = LogisticRegression()
            # Add more classification models as needed
            else:
                continue  # Skip unsupported models
            model_objects[algorithm_name] = model

elif prediction_type == "regression":
    # Create model objects for regression tasks
    for algorithm_name, algorithm_data in data["design_state_data"]["algorithms"].items():
        if algorithm_data["is_selected"]:
            if algorithm_name == "RandomForestRegressor":
                model = RandomForestRegressor()
            elif algorithm_name == "LinearRegression":
                model = LinearRegression()
            # Add more regression models as needed
            else:
                continue  # Skip unsupported models
            model_objects[algorithm_name] = model

# Now, 'model_objects' contains model objects suitable for the specified prediction type.
# You can use these models for training and prediction as needed.
