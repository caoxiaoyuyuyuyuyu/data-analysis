import joblib
import os

import pandas as pd
from flask import current_app
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

    def generate_visualization_data(self, X, y_pred, y_true=None, target_column=None, model_category=None):
        """生成用于前端可视化的数据"""
        viz_data = {
            "model_type": model_category,
            "basic_metrics": None,
            "feature_importance": None,
            "distribution": None,
            "cluster_visualization": None
        }

        try:
            # 1. 基本指标（需要真实标签）
            if y_true is not None:
                if model_category == 'classification':
                    viz_data['class_labels'] = getattr(self.model, 'classes_', sorted(set(y_true))).tolist()
                    viz_data['basic_metrics'] = {
                        'accuracy': accuracy_score(y_true, y_pred),
                        'precision': precision_score(y_true, y_pred, average='weighted'),
                        'recall': recall_score(y_true, y_pred, average='weighted'),
                        'f1': f1_score(y_true, y_pred, average='weighted')
                    }
                elif model_category == 'regression':
                    viz_data['basic_metrics'] = {
                        'mse': mean_squared_error(y_true, y_pred),
                        'r2': r2_score(y_true, y_pred)
                    }

            # 2. 特征重要性
            importance = None
            feature_names = None

            # 处理Pipeline类型模型
            actual_model = self.model
            if hasattr(self.model, 'steps') and isinstance(self.model.steps, list):
                # 获取Pipeline中的最终估计器
                actual_model = self.model.steps[-1][1]

            # 获取特征名称
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            elif hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
            elif hasattr(X, 'shape'):
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            else:
                feature_names = []

            # 检查特征重要性属性
            if hasattr(actual_model, 'feature_importances_'):
                importance = actual_model.feature_importances_
                print(f"使用 feature_importances_ 属性，形状: {importance.shape}")

            # 检查系数属性
            elif hasattr(actual_model, 'coef_'):
                coef = actual_model.coef_

                # 处理不同维度的系数
                if coef.ndim == 1:
                    importance = coef
                elif coef.ndim == 2:
                    # 多分类模型 - 取绝对值平均值
                    importance = np.abs(coef).mean(axis=0)
                    print(f"处理二维 coef_，平均后形状: {importance.shape}")
                else:
                    print(f"不支持的 coef_ 维度: {coef.ndim}")

            # 处理随机森林等集成方法
            elif hasattr(actual_model, 'estimators_'):
                # 尝试从第一个估计器获取特征重要性
                if len(actual_model.estimators_) > 0:
                    first_estimator = actual_model.estimators_[0]
                    if hasattr(first_estimator, 'feature_importances_'):
                        importance = first_estimator.feature_importances_
                        print("使用集成模型中第一个估计器的 feature_importances_")

            # 处理成功获取到重要性值的情况
            if importance is not None and len(importance) > 0:
                # 确保重要性数组长度匹配特征数量
                if len(importance) == len(feature_names):
                    viz_data['feature_importance'] = {
                        'features': feature_names,
                        'importance': importance.tolist()
                    }
                else:
                    print(f"特征数量 ({len(feature_names)}) 与重要性数组长度 ({len(importance)}) 不匹配")
            else:
                print("未找到特征重要性信息")

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