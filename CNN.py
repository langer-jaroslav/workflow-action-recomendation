import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras_tuner import RandomSearch

# -----------------------
# Key Parameters
# -----------------------
FILE_PATH = "./data/requests.csv"         # Path to the dataset file
TEST_SIZE = 0.2                           # Proportion of data for testing
RANDOM_STATE = 42                         # Seed for reproducibility
MAX_TRIALS = 5                            # Number of trials for hyperparameter tuning
EPOCHS = 50                               # Number of epochs for training
BATCH_SIZE = 16                           # Size of the mini-batches during training
PATIENCE = 5                              # Patience for early stopping
MIN_LR = 1e-5                             # Minimum learning rate
INITIAL_LR = 1e-3                         # Initial learning rate
DENSE_LAYER_MIN = 2                       # Minimum number of dense layers
DENSE_LAYER_MAX = 3                       # Maximum number of dense layers
DENSE_UNIT_CHOICES = [128, 256]           # Choices for units in dense layers
DROPOUT_MIN = 0.2                         # Minimum dropout rate
DROPOUT_MAX = 0.5                         # Maximum dropout rate
DROPOUT_STEP = 0.1                        # Step size for dropout rate
# -----------------------

data = pd.read_csv(FILE_PATH)

label_encoder = LabelEncoder()
data['status_encoded'] = label_encoder.fit_transform(data['status'])

X = data[['requested_items', 'is_urgent', 'is_from_wholesaler', 'total_value', 'price_per_item']].copy()
X = pd.get_dummies(X.join(data[['priority', 'order_type']]), drop_first=True)

scaler = StandardScaler()
X[['requested_items', 'total_value', 'price_per_item']] = scaler.fit_transform(X[['requested_items', 'total_value', 'price_per_item']])

X = X.values.astype('float32')
y = data['status_encoded'].values.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

def build_dnn_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X.shape[1],)))
    
    for i in range(hp.Int("dense_layers", DENSE_LAYER_MIN, DENSE_LAYER_MAX)):
        model.add(tf.keras.layers.Dense(
            units=hp.Choice(f"dense_units_{i}", DENSE_UNIT_CHOICES),
            activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(hp.Float("dropout", DROPOUT_MIN, DROPOUT_MAX, step=DROPOUT_STEP)))
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"])
    
    return model

tuner = RandomSearch(
    build_dnn_model,
    objective="val_accuracy",
    max_trials=MAX_TRIALS,
    executions_per_trial=1,
    directory="dnn_tuning",
    project_name="dnn_weighted_flatten")

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=MIN_LR)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, 
             callbacks=[early_stopping, lr_scheduler], class_weight=class_weight_dict)

best_model = tuner.get_best_models(num_models=1)[0]

test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print("Best Test Loss:", test_loss)
print("Best Test Accuracy:", test_accuracy)

y_pred = (best_model.predict(X_test) > 0.5).astype("int32")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("True Positives (TP):", tp)
print("True Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
