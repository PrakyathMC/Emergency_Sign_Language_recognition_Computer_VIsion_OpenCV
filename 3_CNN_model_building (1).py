import os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score, confusion_matrix


import optuna

EPOCHS = 200
N_TRIALS = 10  # Number of Optuna trials

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

def create_cnn_model(input_shape, num_classes, trial=None):
    model = Sequential()
    model.add(
        Conv2D(
            trial.suggest_int("conv_units1", 32, 256),
            kernel_size=(3, 3),
            activation="relu",
            input_shape=input_shape,
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            trial.suggest_int("conv_units2", 32, 256), kernel_size=(3, 3), activation="relu"
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(trial.suggest_int("dense_units1", 64, 512), activation="relu"))
    model.add(Dropout(trial.suggest_float("dropout", 0.2, 0.5)))
    model.add(Dense(trial.suggest_int("dense_units2", 32, 256), activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    return model

def objective(trial):
    input_dir = "Feature_Labels"
    log_dir = "Logs"
    features, labels = load_data(input_dir)
    X, y = prepare_data(features, labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    model = create_cnn_model(input_shape, num_classes, trial)

    model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
    )

    tb_callback = TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=EPOCHS, callbacks=[tb_callback], verbose=1)

    res = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(res, axis=1).tolist()

    acc = accuracy_score(ytrue, yhat)
    precision = precision_score(ytrue, yhat, average='weighted')
    recall = recall_score(ytrue, yhat, average='weighted')
    f1 = f1_score(ytrue, yhat, average='weighted')
    conf_matrix = confusion_matrix(ytrue, yhat)

    # Display the metrics
    print("Accuracy", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    return acc

def main():
    input_dir = "Feature_Labels"
    model_dir = "Models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    features, labels = load_data(input_dir)
    X, y = prepare_data(features, labels)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=N_TRIALS)  # Change number of trials here

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_model = create_cnn_model(X_train.shape[1:], y_train.shape[1], trial)
    best_model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
    )
    best_model.fit(X_train, y_train, epochs=EPOCHS)

    best_model.save(os.path.join(model_dir, "best_cnn_model.h5"))
    print("Best model saved!!")

if __name__ == "__main__":
    main()
