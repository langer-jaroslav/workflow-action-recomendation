import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from keras_tuner import RandomSearch

# Parameters
FILE_PATH = "./data/requests.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_TRIALS = 5
EPOCHS = 50
BATCH_SIZE = 16
PATIENCE = 5
MIN_LR = 1e-5
INITIAL_LR = 1e-3
DENSE_LAYER_MIN = 2
DENSE_LAYER_MAX = 3
DENSE_UNIT_CHOICES = [128, 256]
DROPOUT_MIN, DROPOUT_MAX, DROPOUT_STEP = 0.2, 0.5, 0.1

# Data loading and preprocessing
data = pd.read_csv(FILE_PATH)
label_encoder = LabelEncoder()
data['status_encoded'] = label_encoder.fit_transform(data['status'])
X = pd.get_dummies(data[['requested_items', 'is_urgent', 'is_from_wholesaler', 'total_value', 'price_per_item', 'priority', 'order_type']], drop_first=True)
scaler = StandardScaler()
X[['requested_items', 'total_value', 'price_per_item']] = scaler.fit_transform(X[['requested_items', 'total_value', 'price_per_item']])
X = X.values.astype('float32')
y = data['status_encoded'].values.astype('float32')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Model building function for tuning
def build_dnn_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X.shape[1],)))
    
    # Add dense layers based on hyperparameters
    for i in range(hp.Int("dense_layers", DENSE_LAYER_MIN, DENSE_LAYER_MAX)):
        model.add(tf.keras.layers.Dense(units=hp.Choice(f"dense_units_{i}", DENSE_UNIT_CHOICES), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(hp.Float("dropout", DROPOUT_MIN, DROPOUT_MAX, step=DROPOUT_STEP)))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Hyperparameter tuning
tuner = RandomSearch(
    build_dnn_model, objective="val_accuracy", max_trials=MAX_TRIALS, executions_per_trial=1,
    directory="dnn_tuning", project_name="dnn_weighted_flatten")

# Training with early stopping and learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=MIN_LR)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, 
             callbacks=[early_stopping, lr_scheduler], class_weight=class_weight_dict)

# Select best model and evaluate
best_model = tuner.get_best_models(num_models=1)[0]
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)

# Predictions and classification report
y_pred = (best_model.predict(X_test) > 0.5).astype("int32")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("\nBest Model Hyperparameters:")
best_hps = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values
for param, value in best_hps.items():
    print(f"  {param}: {value}")

# Display all main parameters
print("\nMain Parameters Used:")
print(f"  File path: {FILE_PATH}")
print(f"  Test size: {TEST_SIZE}")
print(f"  Random state: {RANDOM_STATE}")
print(f"  Max trials: {MAX_TRIALS}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Patience for early stopping: {PATIENCE}")
print(f"  Minimum learning rate: {MIN_LR}")
print(f"  Initial learning rate: {INITIAL_LR}")
print(f"  Dense layer min: {DENSE_LAYER_MIN}")
print(f"  Dense layer max: {DENSE_LAYER_MAX}")
print(f"  Dense unit choices: {DENSE_UNIT_CHOICES}")
print(f"  Dropout min: {DROPOUT_MIN}, Dropout max: {DROPOUT_MAX}, Dropout step: {DROPOUT_STEP}")

print(f"\nTest set accuracy: {test_accuracy:.3f}")
print("\nClassification Report:\n", report)
