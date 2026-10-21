# PM-09-PS03: Choose AI Learning Category
import numpy as np 
import keras
from keras import layers
 
print("=" * 60) 
print("PM-09-PS03: CHOOSE AI LEARNING CATEGORY") 
print("=" * 60) 
 
print("\n--- LEARNING CATEGORY COMPARISON ---") 
print(""" 
1. SUPERVISED LEARNING 
   - Data: labelled (input -> known output) 
   - Goal: learn a mapping from input to label 
   - Algorithms: neural networks, decision trees, SVM 
   - Use when: you have historical examples with correct answers 
 
2. UNSUPERVISED LEARNING
   - Data: no dataset; an agent interacts with an environment 
   - Goal: learn a policy that maximises cumulative reward 
   - Algorithms: Q-learning, policy gradients 
   - Use when: sequential decisions with delayed feedback 
""") 

print("--- DATA ANALYSIS FOR CATEGORY SELECTION ---") 
y_train = np.load("../../models/y_train.npy") 
X_train = np.load("../../models/X_train.npy") 
 
print(f"Training features shape: {X_train.shape}") 
print(f"Training labels shape:   {y_train.shape}") 
print(f"Unique label values:     {np.unique(y_train)}") 
print(f"Number of classes:       {len(np.unique(y_train))}") 
 
print("\nDECISION: The data has labelled targets (SalaryBand).") 
print("Therefore, SUPERVISED LEARNING is the correct category.") 
print("Specifically: multi-class classification (3 classes).") 
 
print("\n--- APPLYING SUPERVISED LEARNING ---") 
minimal_model = keras.Sequential([ 
    layers.Input(shape=(X_train.shape[1],)), 
    layers.Dense(8, activation='relu'), 
    layers.Dense(3, activation='softmax') 
]) 
 
minimal_model.compile( 
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'] 
) 
 
minimal_model.summary() 
 
history = minimal_model.fit( 
    X_train, y_train, 
    epochs=3, 
    batch_size=32, 
    verbose=1 
) 

print(f"\nMinimal model trained. Final training accuracy: " 
      f"{history.history['accuracy'][-1]:.4f}") 
 
print("\n--- WHY NOT THE OTHER CATEGORIES? ---") 
print(""" 
Unsupervised: We HAVE labels (SalaryBand), so we don't need 
              to discover structure. We want to predict. 
 
Reinforcement: There is no sequential decision process or 
               reward signal. This is a one-shot prediction. 
""") 
 
print("=" * 60) 
print("Category chosen: SUPERVISED LEARNING (multi-class classification).") 
print("TensorFlow programme created successfully.") 
print("Ready for full model training (PS04).") 
print("=" * 60)
