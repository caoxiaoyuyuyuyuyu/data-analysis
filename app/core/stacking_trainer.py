import joblib
import os
from datetime import datetime
from sklearn.base import is_regressor, is_classifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier, StackingRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier, XGBRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor


# 支持的模型映射表
MODEL_MAP = {
    # 分类模型
    'logistic_regression': LogisticRegression,
    'random_forest': RandomForestClassifier,
    # 'xgboost': XGBClassifier,
    # 'lightgbm': LGBMClassifier,
    'decision_tree': DecisionTreeClassifier,
    'knn': KNeighborsClassifier,
    'svc': SVC,
    'naive_bayes': GaussianNB,
    'ada_boost': AdaBoostClassifier,
    'gradient_boosting': GradientBoostingClassifier,

    # 回归模型
    'linear_regression': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso,
    'svr': SVR,
    'knn_regressor': KNeighborsRegressor,
    'decision_tree_regressor': DecisionTreeRegressor,
    'random_forest_regressor': RandomForestRegressor,
    # 'xgboost_regressor': XGBRegressor,
    # 'lightgbm_regressor': LGBMRegressor,
    'ada_boost_regressor': AdaBoostRegressor,
    'gradient_boosting_regressor': GradientBoostingRegressor,
}

class StackingTrainer:
    def __init__(self, task_type='classification', cv=5):
        self.task_type = task_type
        self.cv = cv
        self.model = None
        self.base_models = []
        self.meta_model = None
        self.metrics = {}

    def set_base_models(self, model_names):
        """设置基模型"""
        for name in model_names:
            if name in MODEL_MAP:
                model_class = MODEL_MAP[name]
                if self.task_type == 'classification' and not is_classifier(model_class):
                    raise ValueError(f"Model {name} is not suitable for classification")
                elif self.task_type == 'regression' and not is_regressor(model_class):
                    raise ValueError(f"Model {name} is not suitable for regression")
                self.base_models.append((name, model_class()))
            else:
                raise ValueError(f"Unsupported model: {name}")

    def set_meta_model(self, model_name):
        """设置元模型"""
        if model_name in MODEL_MAP:
            model_class = MODEL_MAP[model_name]
            if self.task_type == 'classification' and not is_classifier(model_class):
                raise ValueError(f"Meta model {model_name} is not suitable for classification")
            elif self.task_type == 'regression' and not is_regressor(model_class):
                raise ValueError(f"Meta model {model_name} is not suitable for regression")
            self.meta_model = model_class()
        else:
            raise ValueError(f"Unsupported meta model: {model_name}")

    def build_model(self):
        """构建Stacking模型"""
        if self.task_type == 'classification':
            self.model = StackingClassifier(
                estimators=self.base_models,
                final_estimator=self.meta_model,
                cv=self.cv,
                n_jobs=-1
            )
        else:
            self.model = StackingRegressor(
                estimators=self.base_models,
                final_estimator=self.meta_model,
                cv=self.cv,
                n_jobs=-1
            )

    def train(self, X_train, y_train):
        """执行训练"""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """模型评估"""
        preds = self.model.predict(X_test)
        if self.task_type == 'classification':
            self.metrics['accuracy'] = accuracy_score(y_test, preds)
        else:
            self.metrics['mse'] = mean_squared_error(y_test, preds)
        return self.metrics

    def save_model(self, model_dir='models'):
        """保存模型文件"""
        os.makedirs(model_dir, exist_ok=True)
        self.model_id = f"stacking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_path = os.path.join(model_dir, f"{self.model_id}.pkl")
        joblib.dump(self.model, self.model_path)
        return self.model_id, self.model_path