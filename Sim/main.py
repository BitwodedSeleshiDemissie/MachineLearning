def main():
    from pathlib import Path
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import pandas as pd
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    import numpy as np


    # Resolve CSV relative to this file so it works no matter the working directory
    data_path = Path(__file__).with_name("train_dataset.csv")
    dataset = pd.read_csv(data_path)

    # Redefine feature_columns to exclude 'target'
    feature_columns = ['cont_0', 'cont_1', 'cont_2', 'cont_3', 'cont_4', 'cont_5', 'cont_6',
        'cont_7', 'cont_8', 'cont_9', 'cont_10', 'cont_11', 'cont_12',
        'cont_13', 'cont_14', 'cont_15', 'cont_16', 'cont_17', 'cont_18',
        'cont_19', 'cont_20', 'cont_21', 'cont_22', 'cont_23', 'cont_24',
        'cont_25', 'cont_26', 'cont_27', 'cont_28', 'cont_29', 'ord_0', 'ord_1',
        'ord_2', 'ord_3', 'ord_4', 'ord_5', 'ord_6', 'ord_7', 'ord_8', 'ord_9',
        'ord_10', 'ord_11', 'ord_12', 'ord_13', 'ord_14', 'ord_15', 'ord_16',
        'ord_17', 'ord_18', 'ord_19', 'cat_0', 'cat_1', 'cat_2', 'cat_3',
        'cat_4', 'cat_5', 'cat_6', 'cat_7'] # 'target' removed from here

    target = ['target'] # Already correctly defined
    X = dataset[feature_columns]
    y = dataset[target]


    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state= 42)


    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(X_train) # Imputer now fitted only on feature columns
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    # The X_train_imputed is a numpy array that still contains string values
    # for categorical/ordinal features. LinearRegression expects numerical input.
    # We need to encode these string features.

    # First, identify the types of features based on the `feature_columns` list
    # These lists are already correctly defined from the previous steps.
    continuous_features = [f'cont_{i}' for i in range(30)]
    ordinal_features = [f'ord_{i}' for i in range(20)]
    categorical_features = [f'cat_{i}' for i in range(8)]

    # Convert the numpy array back to a DataFrame to use ColumnTransformer with column names
    # This ensures the ColumnTransformer correctly identifies and processes columns.
    # `feature_columns` is available in the kernel state and maintains the original order.
    processed_X_train = pd.DataFrame(X_train_imputed, columns=feature_columns)
    processed_X_test = pd.DataFrame(X_test_imputed, columns=feature_columns)

    # Create a ColumnTransformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', continuous_features), # Keep continuous features as they are
            # One-hot encode ordinal and categorical features
            # handle_unknown='ignore' will prevent errors if a category not seen during fit is encountered during transform
            ('ord', OneHotEncoder(handle_unknown='ignore'), ordinal_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # This should ideally not be needed if all columns are specified
    )

    # Fit the preprocessor on the training data and transform both training and test data
    X_train_encoded = preprocessor.fit_transform(processed_X_train)
    X_test_encoded = preprocessor.transform(processed_X_test)

    model = LinearRegression()
    model.fit(X_train_encoded, y_train)
    print(model)
    y_pred = model.predict(X_test_encoded)

    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R-squared: {r_squared:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
if __name__ == '__main__':
    main()
