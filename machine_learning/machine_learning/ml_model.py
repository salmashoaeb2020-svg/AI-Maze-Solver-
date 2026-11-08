import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from machine_learning.dataset_generator import generate_ml_dataset

class MazeHeuristicModel:
    """
    Wrapper for Scikit-learn Decision Tree Regressor to predict remaining distance.
    """
    def __init__(self):
        self.model = DecisionTreeRegressor(random_state=42)
        self.mae = 0.0

    def train(self, X=None, y=None):
        if X is None or y is None:
            X, y = generate_ml_dataset(num_mazes=10, width=10, height=10)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        self.mae = mean_absolute_error(y_test, predictions)
        return self.mae

    def predict_heuristic(self, pos, goal, width, height):
        r, c = pos
        gr, gc = goal
        manh = abs(r - gr) + abs(c - gc)
        features = np.array([[r, c, gr, gc, manh, width, height]])
        pred = self.model.predict(features)
        return max(0.0, pred[0])
