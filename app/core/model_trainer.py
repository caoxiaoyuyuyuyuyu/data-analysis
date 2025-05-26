from random import randint, uniform

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingClassifier, AdaBoostRegressor, \
    GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split, learning_curve, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score, explained_variance_score, mean_absolute_error
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


class ModelTrainer:
    """增强版模型训练与评估类"""

    def __init__(self):
        self.models = {
            'regression': {
                'Linear Regression': LinearRegression(),
                'Polynomial Regression': None,  # 特殊处理
                'Ridge Regression': Ridge(),
                'Lasso Regression': Lasso(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'SVR': SVR(),
                'KNN Regression': KNeighborsRegressor()
            },
            'classification': {
                'Logistic Regression': LogisticRegression(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'SVM': SVC(probability=True),
                'KNN Classification': KNeighborsClassifier()
            },
            'clustering': {
                'K-Means': KMeans(),
                'PCA': PCA()
            }
        }
        self.best_model = None
        self.best_score = 0
        self.scaler = None

    def determine_problem_type(self, y):
        """更健壮的问题类型检测"""
        y_array = np.array(y)
        unique_values = np.unique(y_array)

        if len(unique_values) <= 10 or y_array.dtype.kind in ['O', 'U', 'S']:
            return 'classification'
        return 'regression'

    def create_pipeline(self, model_name, normalize=False, degree=2):
        """创建处理管道"""
        steps = []

        if normalize:
            steps.append(('scaler', StandardScaler()))

        if model_name == 'Polynomial Regression':
            steps.append(('poly', PolynomialFeatures(degree=degree)))
            steps.append(('linear', LinearRegression()))
        else:
            steps.append(('model', self.models[self.current_problem_type][model_name]))

        return Pipeline(steps)

    def train_model(self, X, y, model_name, test_size=0.2, random_state=42,
                    normalize=False, **params):
        """训练单个模型"""
        self.current_problem_type = self.determine_problem_type(y)

        if model_name not in self.models[self.current_problem_type]:
            raise ValueError(f"模型 {model_name} 不适用于 {self.current_problem_type} 问题")

        # 创建管道
        pipeline = self.create_pipeline(model_name, normalize, params.get('degree', 2))

        # 设置参数
        if model_name == 'Polynomial Regression':
            pipeline.set_params(**{'poly__degree':params.get('degree', 2)})
        else:
            pipeline.set_params(**{f'model__{k}': v for k, v in params.items()})

        # 划分数据集
        if self.current_problem_type in ['regression', 'classification']:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = self.evaluate_model(y_test, y_pred, self.current_problem_type)

            return {
                'model': pipeline,
                'metrics': metrics,
                'X_test': X_test,
                'y_test': y_test,
                'learning_curve': self.get_learning_curve(pipeline, X, y)
            }
        else:  # 聚类或无监督学习
            pipeline.fit(X)

            if model_name == 'K-Means':
                score = silhouette_score(X, pipeline.predict(X))
                metrics = {'silhouette_score': score}
            else:  # PCA
                metrics = {'explained_variance': pipeline.explained_variance_ratio_}

            return {
                'model': pipeline,
                'metrics': metrics,
                'X_transformed': pipeline.transform(X)
            }

    def evaluate_model(self, y_true, y_pred, problem_type):
        """评估模型性能"""
        metrics = {}

        if problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        else:  # regression
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)

        return metrics

    def get_learning_curve(self, model, X, y, cv=5):
        """获取学习曲线数据"""
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5))
        #
        # return {
        #     'train_sizes': train_sizes,
        #     'train_scores': train_scores.mean(axis=1),
        #     'test_scores': test_scores.mean(axis=1)
        # }
        return {
            'train_sizes': train_sizes.tolist(),  # 转换为列表
            'train_scores': train_scores.mean(axis=1).tolist(),  # 转换为列表
            'test_scores': test_scores.mean(axis=1).tolist()  # 转换为列表
        }

    def get_cross_val_scores(self, X, y, model_name, params=None, cv=5, scoring=None):
        """获取交叉验证分数用于箱线图比较"""
        if params is None:
            params = {}

        problem_type = self.determine_problem_type(y)
        model = self.models[problem_type][model_name].set_params(**params)

        if problem_type == 'classification':
            if scoring is None:
                scoring = 'accuracy'
        else:  # regression
            if scoring is None:
                scoring = 'r2'

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        return scores