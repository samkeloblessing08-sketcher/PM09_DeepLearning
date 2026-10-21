# PM-09-PS04: Train Mode
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.callbacks import EarlyStopping 
import matplotlib.pyplot as plt 
import joblib 
from sklearn.metrics import classification_report 
 
print("=" * 60) 
print("PM-09-PS04: TRAIN MODEL") 
print("=" * 60) 
 
# 1. LOAD PREPARED DATA 
X_train = np.load("../../models/X_train.npy") 
X_val = np.load("../../models/X_val.npy") 
X_test = np.load("../../models/X_test.npy") 
y_train = np.load("../../models/y_train.npy") 
y_val = np.load("../../models/y_val.npy") 
y_test = np.load("../../models/y_test.npy") 
label_encoder = joblib.load("../../models/label_encoder.pkl") 
 
print(f"Training:   {X_train.shape}") 
print(f"Validation: {X_val.shape}") 
print(f"Test:       {X_test.shape}") 
 
n_features = X_train.shape[1] 
n_classes = len(np.unique(y_train)) 
 
# 2. MODEL SELECTION 
print("\n--- MODEL SELECTION ---") 
print(""" 
Problem type: Multi-class classification (3 classes) 
Data type:    Tabular, numeric, ~13 features, 350 training rows 
 
Chosen architecture: Feedforward Neural Network (MLP) 
  - Hidden layer 1: 32 neurons, ReLU 
  - Hidden layer 2: 16 neurons, ReLU 
  - Output layer: 3 neurons, softmax 
 
Why this architecture? 
  - Tabular data with non-linear interactions -> MLP works well 
  - Small dataset -> keep the network small to avoid overfitting 
""") 
 
# 3. ALGORITHM CHOICE 
print("--- ALGORITHM CHOICE ---") 
print(""" 
Learning category: Supervised learning 
Loss function:     Sparse categorical crossentropy 
Optimizer:         Adam (adaptive learning rate) 
Metric:            Accuracy 
Regularisation:    Dropout + EarlyStopping 
""") 
 
# 4. BUILD THE MODEL 
def build_model(n_features, n_classes): 
    model = keras.Sequential([ 
        layers.Input(shape=(n_features,)), 
        layers.Dense(32, activation='relu', name='hidden_1'), 
        layers.Dropout(0.3, name='dropout_1'), 
        layers.Dense(16, activation='relu', name='hidden_2'), 
        layers.Dropout(0.2, name='dropout_2'), 
        layers.Dense(n_classes, activation='softmax', name='output') 
    ]) 
    return model 
 
model = build_model(n_features, n_classes) 
model.summary() 
 
# 5. COMPILE 
model.compile( 
    optimizer=keras.optimizers.Adam(learning_rate=0.001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'] 
) 
 
early_stop = EarlyStopping( 
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1 
) 
 
# 6. TRAIN 
print("\n--- TRAINING ---") 
history = model.fit( 
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=150, 
    batch_size=32, 
    callbacks=[early_stop], 
    verbose=1 
) 
 
# 7. EVALUATE 
print("\n--- EVALUATION ---") 
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0) 
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0) 
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0) 
 
print(f"Train accuracy:      {train_acc:.4f}") 
print(f"Validation accuracy: {val_acc:.4f}") 
print(f"Test accuracy:       {test_acc:.4f}") 
 
print(f"\nPerformance standard: >= 0.80 test accuracy") 
print(f"Result: {'PASS' if test_acc >= 0.80 else 'FAIL'}") 
 
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1) 
 
print("\n--- PER-CLASS REPORT ---") 
print(classification_report( 
    y_test, y_pred, 
    target_names=label_encoder.classes_ 
)) 
 
# 8. PREDICTABILITY CHECK 
print("--- MODEL PREDICTABILITY CHECK ---") 
print("Running the same test set through the model 3 times.") 
print("A stable model should give identical predictions each time.\n") 
 
for i in range(3): 
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1) 
    acc = (preds == y_test).mean() 
    print(f"Run {i+1}: test accuracy = {acc:.4f}") 
 
# 9. LEARNING CURVES 
fig, axes = plt.subplots(1, 2, figsize=(12, 4)) 
 
axes[0].plot(history.history['loss'], label='Train loss') 
axes[0].plot(history.history['val_loss'], label='Val loss') 
axes[0].set_title('Loss over epochs') 
axes[0].set_xlabel('Epoch') 
axes[0].set_ylabel('Loss') 
axes[0].legend() 
 
axes[1].plot(history.history['accuracy'], label='Train accuracy') 
axes[1].plot(history.history['val_accuracy'], label='Val accuracy') 
axes[1].set_title('Accuracy over epochs') 
axes[1].set_xlabel('Epoch') 
axes[1].set_ylabel('Accuracy') 
axes[1].legend() 
 
plt.tight_layout() 
plt.savefig("../../evidence/04a_learning_curves.png", dpi=100) 
plt.show() 
 
# 10. SAVE THE TRAINED MODEL 
model.save("../../models/hr_model.keras") 
print("\nModel saved to models/hr_model.keras") 
 
print("\n" + "=" * 60) 
print("Model trained and evaluated.") 
print("Ready for ML system selection (PS05).") 
print("=" * 60)
