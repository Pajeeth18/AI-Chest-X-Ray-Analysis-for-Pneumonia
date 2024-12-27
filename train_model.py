import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from PIL import Image
import joblib


TRAIN_DIR = 'C:\\Jupyter\\Medical_Diagnosis\\data\\train'
MODEL_PATH = 'C:\\Jupyter\\Medical_Diagnosis\\models\\pneumonia_model.pkl'

def load_images_and_labels(directory):
    images = []
    labels = []
    for label, class_name in enumerate(['normal', 'pneumonia']):
        class_dir = os.path.join(directory, class_name)
        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)
            try:
                img = Image.open(img_path).resize((150, 150)).convert('L')
                img_array = np.array(img).flatten()  
                images.append(img_array)
                labels.append(label)  
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels)

print("Loading training data...")
X, y = load_images_and_labels(TRAIN_DIR)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['Normal', 'Pneumonia']))

os.makedirs('models', exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
