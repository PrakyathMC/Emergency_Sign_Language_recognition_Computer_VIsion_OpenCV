import os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score
import optuna

EPOCHS = 20
N_TRIALS = 20 # Number of optuna trials

def load_data(input_dir):
    features = np.load(os.path.join(input_dir, 'features.npy'))
    labels = np.load(os.path.join(input_dir, 'labels.npy'))
    return features, labels

def prepare_data(features, labels):
    unique_labels = sorted(set(labels))
    label_map = {label: num for num, label in enumerate(unique_labels)}
    labels_num = np.array([label_map[label] for label in labels])

    X = np.array(features)
    y = to_categorical(labels_num).astype(int)
    
    return X, y

# def create_lstm_model(input_shape, num_classes, trial=None):
#     model = Sequential()
#     model.add(LSTM(trial.suggest_int("lstm_units1", 32, 256), input_shape=input_shape, return_sequences=True))
#     model.add(LSTM(trial.suggest_int("lstm_units2", 32, 256), return_sequences=False))
#     model.add(Dense(trial.suggest_int("dense_units1", 64, 512), activation='relu'))
#     model.add(Dropout(trial.suggest_float("dropout", 0.2, 0.5)))
#     model.add(Dense(trial.suggest_int("dense_units2", 32, 256), activation='relu'))
#     model.add(Dense(num_classes, activation='softmax'))
#     return model

def create_lstm_model(input_shape, num_classes, trial=None):
    model = Sequential()
    lstm_units1 = trial.suggest_int("lstm_units1", 64, 256)
    model.add(LSTM(lstm_units1, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(trial.suggest_float("dropout1", 0.2, 0.5)))

    lstm_units2 = trial.suggest_int("lstm_units2", 64, 256)
    model.add(LSTM(lstm_units2))
    model.add(Dropout(trial.suggest_float("dropout2", 0.2, 0.5)))
    model.add(Dense(trial.suggest_int("dense_units1", 64, 256), activation="relu"))
    model.add(Dropout(trial.suggest_float("dropout3", 0.2, 0.5)))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def objective(trial):
    input_dir = 'Feature_Labels'
    log_dir = 'Logs'
    features, labels = load_data(input_dir)
    X, y = prepare_data(features, labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    
    model = create_lstm_model(input_shape, num_classes, trial)
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    tb_callback = TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=EPOCHS, callbacks=[tb_callback], verbose=1)
    
    res = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(res, axis=1).tolist()
    acc = accuracy_score(ytrue, yhat)
    #f1 = f1_score(ytrue, yhat, average='samples')
    return acc

def main():
    input_dir = 'Feature_Labels'
    model_dir = 'Models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    features, labels = load_data(input_dir)
    X, y = prepare_data(features, labels)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS) # Change number of trials here
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    best_model = create_lstm_model(X_train.shape[1:], y_train.shape[1], trial)
    best_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    best_model.fit(X_train, y_train, epochs=EPOCHS)
    
    best_model.save(os.path.join(model_dir, "best_lstm_model.h5"))
    print("Best model saved!!")

if __name__ == "__main__":
    main()