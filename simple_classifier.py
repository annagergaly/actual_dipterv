import os
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tsfresh import select_features


PREPROCESS_TYPE = f'Outputs/cpac/filt_noglobal/rois_ho'
ROOT_PATH = f'C:/Users/Anna/Documents/actual_dipterv/data'


def load_site(site):
    autistic = []
    control = []
    for roi_file in os.listdir(f'{ROOT_PATH}/{site}/{PREPROCESS_TYPE}'):
        data = np.loadtxt(f'{ROOT_PATH}/{site}/{PREPROCESS_TYPE}/{roi_file}')
        autistic.append(data)

    for roi_file in os.listdir(f'{ROOT_PATH}/{site}_control/{PREPROCESS_TYPE}'):
        data = np.loadtxt(f'{ROOT_PATH}/{site}_control/{PREPROCESS_TYPE}/{roi_file}')
        control.append(data)

    return (autistic, control)


def prepare_data(autistic, control):
    a = np.ones(len(autistic))
    c = np.zeros(len(control))
    print(a)
    print(c)
    labels = np.concat((a, c))
    #autistic = np.stack(autistic)
    #control = np.stack(control)

    df_list = []
    for i, seq in enumerate(autistic + control):
        df = pd.DataFrame(seq)
        df['id'] = i
        df['time'] = np.arange(len(seq))
        df_list.append(df)

    df_long = pd.concat(df_list, ignore_index=True)

    df_features = extract_features(df_long, column_id="id", column_sort="time", default_fc_parameters=EfficientFCParameters())

    X_extracted = select_features(df_features, labels)
    X_extracted = df_features.to_numpy()
    X = np.concat((autistic, control))
    X = X.reshape(X.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(X_extracted, labels, test_size=0.2, random_state=42)

    return (X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    print('NYU')
    (autistic, control) = load_site('NYU')
    (X_train, X_test, y_train, y_test) = prepare_data(autistic, control)

    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)

    # Evaluate
    y_pred = xgb_model.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf_model.fit(X_train, y_train)

    # y_pred = rf_model.predict(X_test)
    # print(f"Random forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # svm_model = SVC(kernel="rbf", probability=True, random_state=42)
    # svm_model.fit(X_train, y_train)

    # y_pred = svm_model.predict(X_test)
    # print(f"SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# for roi_file in os.listdir(f'{ROOT_PATH}/KKI/{PREPROCESS_TYPE}'):
#     data = np.loadtxt(f'{ROOT_PATH}/KKI/{PREPROCESS_TYPE}/{roi_file}')
#     print(data.shape)
    