import pandas as pd
from ydata_profiling import ProfileReport
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score
np.random.seed(42)



def open_data(path="data.csv"):
    df = pd.read_csv(path)
    return df

def recode_value(value):
    if (value == 'Москва') | (value == 'Санкт-Петербург'):
        return 1
    else:
        return 0

def split_data(df: pd.DataFrame):
    y = df['Survived']
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

    return X, y

def preprocess_data(df: pd.DataFrame, test=True):
    df.drop(['WORK_TIME'], axis = 1,  inplace = True)
    df['FACT_ADDRESS_PROVINCE'] = df['FACT_ADDRESS_PROVINCE'].apply(recode_value)

    if test:
        X_train, X_test, y_train, y_test = train_test_split(df.drop(['TARGET'], axis = 1), df['TARGET'], test_size=0.2, random_state=42)
    else:
        X_df = df
    

    report = ProfileReport(df)

    if test:
        return X_train, X_test, y_train, y_test
    else:
        return X_df


def fit_and_save_model(X_train, X_test, y_train, y_test, path="data/model_weights.mw"):
    sc = StandardScaler() # x -> (x - mean) / std
    sc.fit(X_train) # вычисляем mean, std

    X_train = pd.DataFrame(sc.transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns = X_test.columns)
    
    lr = LogisticRegression() # точное решение
    lr.fit(X_train, y_train) # минимизируем функцию потерь
    y_pred_lr = lr.predict_proba(X_test)
    y_pred = np.argmax(y_pred_lr > 0.5, axis = 1)
    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Model accuracy is {accuracy}")

    #with open(path, "wb") as file:
    #    dump(model, file)

    #print(f"Model was saved to {path}")
    return accuracy

def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Вам не повезло с вероятностью",
        1: "Вы выживете с вероятностью"
    }

    encode_prediction = {
        0: "Сожалеем, вам не повезло",
        1: "Ура! Вы будете жить"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    fit_and_save_model(X_train, X_test, y_train, y_test)