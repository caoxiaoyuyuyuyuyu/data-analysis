import joblib
import os

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score
)

class ModelPredictor:

    def __init__(self, model_path: str, model_type: str = None):
        """
        初始化预测器，加载模型并识别其类型

        :param model_path: 模型文件路径（.pkl 文件）
        :param model_type: 模型类型（'regression', 'classification', 'clustering'）
        :raises: FileNotFoundError, ValueError
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"加载模型失败: {e}")

        if not isinstance(self.model, BaseEstimator):
            raise ValueError("加载的模型必须是 sklearn 的 BaseEstimator 子类")

        # 设置模型类型
        valid_types = ['regression', 'classification', 'clustering']
        if model_type and model_type.lower() in valid_types:
            self.model_type = model_type.lower()
        else:
            raise NotImplementedError("模型类型不支持")


    def predict(self, X):
        """
        对输入数据进行预测

        :param X: 输入数据，可以是 DataFrame 或 numpy array
        :return: 预测结果
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        对分类模型进行概率预测

        :param X: 输入数据
        :return: 概率预测结果
        :raises: 如果模型不支持 predict_proba 方法
        """
        if self.model_type != 'classification':
            raise NotImplementedError("非分类模型不支持 predict_proba 方法")
        if hasattr(self.model, 'predict_proba') and callable(self.model.predict_proba):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("该模型不支持 predict_proba 方法")

    def generate_visualization_data(self, X, y_pred, target_column, model_category):
        """
        生成用于前端可视化的数据
        """
        viz_data = {
            "model_type": model_category,
            "basic_metrics": None,
            "feature_importance": None,
            "distribution": None,
            "cluster_visualization": None
        }

        try:
            # 1. 基本指标
            if model_category == 'classification' and hasattr(self.model, 'classes_'):
                viz_data['class_labels'] = self.model.classes_.tolist()

            # 2. 特征重要性
            if hasattr(self.model, 'feature_importances_'):
                viz_data['feature_importance'] = {
                    'features': X.columns.tolist(),
                    'importance': self.model.feature_importances_.tolist()
                }
            elif hasattr(self.model, 'coef_'):
                viz_data['feature_importance'] = {
                    'features': X.columns.tolist(),
                    'importance': self.model.coef_.tolist()
                }

            # 3. 预测结果分布
            if model_category in ['classification', 'regression']:
                viz_data['distribution'] = {
                    'predicted': pd.Series(y_pred).value_counts().to_dict()
                }
                if target_column in X.columns:  # 如果有真实值可以对比
                    viz_data['distribution']['actual'] = X[target_column].value_counts().to_dict()

            # 4. 聚类可视化
            if model_category == 'clustering':
                # 使用PCA或TSNE降维
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(X)
                viz_data['cluster_visualization'] = {
                    'x': reduced_data[:, 0].tolist(),
                    'y': reduced_data[:, 1].tolist(),
                    'labels': y_pred.tolist(),
                    'cluster_centers': getattr(self.model, 'cluster_centers_', None)
                }

        except Exception as e:
            raise
        return viz_data