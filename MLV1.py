# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from scipy import stats

# -------------------- Загрузка Данных --------------------
print("=" * 50)
print("Загрузка Данных")
print("=" * 50)

# Загрузка датасета ирисов
iris = load_iris(as_frame=True)
data = iris.frame
print("Первые 5 строк данных:")
print(data.head())
print("\nИнформация о данных:")
print(data.info())
print("\nСтатистическое описание данных:")
print(data.describe())
print("\nПроверка на пропущенные значения:")
print(data.isnull().sum())

# -------------------- Визуализация Данных --------------------
print("\n" + "=" * 50)
print("Визуализация Данных")
print("=" * 50)

# Гистограммы
data.hist(bins=20, figsize=(15, 10))
plt.suptitle("Распределение признаков", fontsize=16)
plt.show()

# Точечные диаграммы
sns.pairplot(data, hue='target', palette='viridis')
plt.suptitle("Взаимосвязи между признаками", fontsize=16)
plt.show()

# Корреляционная матрица
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Корреляционная матрица")
plt.show()

# Распределение целевой переменной
sns.countplot(x='target', data=data)
plt.title("Распределение видов ирисов")
plt.show()

# Box plots
plt.figure(figsize=(12, 8))
for i, column in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='target', y=column, data=data, palette='Set2')
plt.suptitle("Box plots признаков по видам ирисов", fontsize=16)
plt.tight_layout()
plt.show()

# Violin plots
plt.figure(figsize=(12, 8))
for i, column in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.violinplot(x='target', y=column, data=data, palette='Set2')
plt.suptitle("Violin plots признаков по видам ирисов", fontsize=16)
plt.tight_layout()
plt.show()

# -------------------- Предварительная Обработка Данных --------------------
print("\n" + "=" * 50)
print("Предварительная Обработка Данных")
print("=" * 50)

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Размер обучающей выборки:", X_train_scaled.shape)
print("Размер тестовой выборки:", X_test_scaled.shape)

# -------------------- Построение Моделей Машинного Обучения --------------------
print("\n" + "=" * 50)
print("Построение Моделей Машинного Обучения")
print("=" * 50)

# --- Scikit-learn ---
print("\n--- Scikit-learn ---")
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{name}:")
    print(f"  Точность: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Фактические значения')
    plt.title(f'Матрица ошибок для {name}')
    plt.show()

    results.append({'Model': name, 'Accuracy': accuracy, 'Report': report})

# --- Keras/TensorFlow ---
print("\n--- Keras/TensorFlow ---")
model_tf = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model_tf.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_tf = model_tf.fit(X_train_scaled, y_train, epochs=100, verbose=0)
loss_tf, accuracy_tf = model_tf.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Нейронная сеть (TensorFlow):")
print(f"  Точность: {accuracy_tf:.4f}")
results.append({'Model': 'TensorFlow NN', 'Accuracy': accuracy_tf})

# График обучения TensorFlow
plt.figure(figsize=(10, 5))
plt.plot(history_tf.history['accuracy'])
plt.title('Точность модели (TensorFlow)')
plt.ylabel('Точность')
plt.xlabel('Эпоха')
plt.show()

# --- PyTorch ---
print("\n--- PyTorch ---")

class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)  # Изменено здесь

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = IrisDataset(X_train_scaled, y_train)
test_dataset = IrisDataset(X_test_scaled, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

input_size = X_train_scaled.shape[1]
output_size = len(np.unique(y))
model_pt = SimpleNN(input_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pt.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Прямой проход
        outputs = model_pt(inputs)
        loss = criterion(outputs, labels)

        # Обратное распространение и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Вывод информации о процессе обучения
    if (epoch+1) % 20 == 0:
        print (f'Эпоха [{epoch+1}/{epochs}], Шаг [{i+1}/{len(train_loader)}], Потери: {loss.item():.4f}')

# Оценка модели PyTorch
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model_pt(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy_pt = correct / total
print(f"Нейронная сеть (PyTorch):")
print(f"  Точность на тестовой выборке: {accuracy_pt:.4f}")
results.append({'Model': 'PyTorch NN', 'Accuracy': accuracy_pt})

# -------------------- Применение SciPy для Статистического Анализа --------------------
print("\n" + "=" * 50)
print("Применение SciPy для Статистического Анализа")
print("=" * 50)

# Проведем ANOVA-тест для сравнения средних значений 'sepal length (cm)' между видами
feature = 'sepal length (cm)'
f_statistic, p_value = stats.f_oneway(
    data[data['target'] == 0][feature],
    data[data['target'] == 1][feature],
    data[data['target'] == 2][feature]
)

print(f"F-статистика для сравнения средних значений '{feature}' между видами ирисов: {f_statistic:.4f}")
print(f"P-значение: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Отвергаем нулевую гипотезу: существует статистически значимая разница в средних значениях данного признака между видами ирисов.")
else:
    print("Не отвергаем нулевую гипотезу: недостаточно доказательств для утверждения о статистически значимой разнице.")

# -------------------- Сохранение результатов --------------------
print("\n" + "=" * 50)
print("Сохранение результатов")
print("=" * 50)

results_df = pd.DataFrame(results)
print("\nРезультаты работы моделей:")
print(results_df)

results_df.to_csv('model_evaluation_results.csv', index=False)
print("\nРезультаты сохранены в файл 'model_evaluation_results.csv'")

print("\n" + "=" * 50)
print("Конец программы")
print("=" * 50)