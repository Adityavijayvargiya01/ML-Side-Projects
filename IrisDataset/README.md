# Iris Dataset Machine Learning Project

This project demonstrates the use of the Iris dataset in a machine learning workflow. The steps include data loading, preprocessing, visualization, model training, and evaluation using Python libraries such as Pandas, NumPy, Matplotlib, and scikit-learn.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Steps](#project-steps)
- [Visualization](#visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [License](#license)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/iris-ml-project.git
    cd iris-ml-project
    ```

2. Install the required Python libraries:

    ```sh
    pip install pandas numpy matplotlib scikit-learn
    ```

## Usage

1. Open the `iris_ml_project.py` script in PyCharm or any other Python IDE.
2. Run the script to see the data loading, preprocessing, visualization, model training, and evaluation steps.

Alternatively, you can use a Jupyter Notebook:

1. Open the `iris_ml_project.ipynb` notebook in Jupyter.
2. Run the cells step-by-step to see the results interactively.

## Project Steps

1. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    ```

2. **Load the Iris Dataset**:
    ```python
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    ```

3. **Convert to a DataFrame**:
    ```python
    df = pd.DataFrame(data=np.c_[X, y], columns=iris.feature_names + ['target'])
    df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    print(df.head())
    ```

4. **Visualize the Data**:
    ```python
    plt.figure(figsize=(12, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Iris Dataset: Sepal Length vs Sepal Width')
    plt.show()
    ```

5. **Split the Data into Training and Test Sets**:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    ```

6. **Standardize the Features**:
    ```python
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    ```

7. **Apply PCA (Principal Component Analysis)**:
    ```python
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    ```

8. **Visualize the PCA-transformed Data**:
    ```python
    plt.figure(figsize=(12, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Iris Dataset')
    plt.show()
    ```

## Model Training and Evaluation

1. **Train a K-Nearest Neighbors Classifier**:
    ```python
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)
    ```

2. **Make Predictions**:
    ```python
    y_pred = knn.predict(X_test_pca)
    ```

3. **Evaluate the Model**:
    ```python
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print(confusion_matrix(y_test, y_pred))
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
