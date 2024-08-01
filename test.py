# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns

# # Load the trained model
# model = tf.keras.models.load_model('best_model.keras')

# # Directories
# valid_dir = 'dataset/valid'

# # Data Generator
# valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# valid_generator = valid_datagen.flow_from_directory(
#     valid_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=False
# )

# # Evaluate the model
# loss, accuracy = model.evaluate(valid_generator)
# print(f'Validation Loss: {loss}')
# print(f'Validation Accuracy: {accuracy}')

# # Predictions
# predictions = model.predict(valid_generator)
# predicted_classes = np.argmax(predictions, axis=1)
# true_classes = valid_generator.classes
# class_labels = list(valid_generator.class_indices.keys())

# # Classification report
# report = classification_report(true_classes, predicted_classes, target_names=class_labels)
# print(report)

# # Confusion matrix
# cm = confusion_matrix(true_classes, predicted_classes)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix')
# plt.show()



###
#new one
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model
model = load_model('best_model.keras')

# Data generators for validation
valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_directory(
    'dataset/valid',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
evaluation = model.evaluate(valid_generator)
print(f"Evaluation Metrics: {evaluation}")

# Get the predictions
predictions = model.predict(valid_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_generator.classes
class_labels = list(valid_generator.class_indices.keys())

# Plotting accuracy and loss
history_dict = model.history.history

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print('Classification Report')
print(report)

# Additional Evaluation Metrics
accuracy = evaluation[1]
precision = tf.keras.metrics.Precision()(true_classes, predicted_classes)
recall = tf.keras.metrics.Recall()(true_classes, predicted_classes)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

