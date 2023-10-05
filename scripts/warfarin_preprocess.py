from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import csv
import pandas as pd

def preprocess_data(binary_label=True):
    #######################################################
    # Load the feature header
    #######################################################
    # read the csv file
    feature_names = []
    with open('./data/header.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            feature_names.append(line)

    # Dirty tricks to clean up the feature names
    feature_names = feature_names[0]
    feature_names = list(map(lambda x: x.replace(' ', '_'), feature_names))
    feature_names[0] = 'PharmGKB_Subject_ID' # manual fix to remove a character in f...[0]

    ####################################################################
    # Load the dataframe into X; clean up some unnecessary columns 
    ####################################################################

    # Load data and process column names
    df = pd.read_csv('./data/warfarin_with_dose.csv', names = feature_names)
    #keep_default_na=False)
    df.columns = [c.replace(' ', '_') for c in df.columns]

    # Extract the label (Warfarin dose) from the rest of the features
    y = df.Therapeutic_Dose_of_Warfarin
    X = df.drop('Therapeutic_Dose_of_Warfarin',axis=1)
    feature_names.remove('Therapeutic_Dose_of_Warfarin')

    # Drop Subject_ID (irrelevant) and Medication (different to encode)
    X = X.drop('PharmGKB_Subject_ID',axis=1)
    X = X.drop('Medications',axis=1)
    X = X.drop('Comorbidities',axis=1)
    feature_names.remove('PharmGKB_Subject_ID')
    feature_names.remove('Medications')
    feature_names.remove('Comorbidities')

    ####################################################################
    # Encode different features with numeric/label/onehot encodings 
    ####################################################################
    numeric_features = ['Height_(cm)',
                        'Weight_(kg)',
                        'Target_INR',
                        'INR_on_Reported_Therapeutic_Dose_of_Warfarin',
                        ]
    label_features =   ['Age',
                        'Estimated_Target_INR_Range_Based_on_Indication']
    categorical_features = [f for f in feature_names \
                            if f not in numeric_features and f not in label_features]

    for feat in categorical_features:
        X[feat] = X[feat].astype(str)

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    label_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='0-missing')),
        ('ordinal', OrdinalEncoder())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('lab', label_transformer, label_features),
            ('cat', categorical_transformer, categorical_features)]
    )

    X_transformed = preprocessor.fit_transform(X)

    if binary_label:
        y = (y>30).astype(int)

    return X_transformed, y
