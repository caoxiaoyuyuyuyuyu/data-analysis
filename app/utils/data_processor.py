import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats


class DataProcessor:
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据集的统计摘要"""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_stats": DataProcessor._get_numeric_stats(df),
            "categorical_stats": DataProcessor._get_categorical_stats(df)
        }

    @staticmethod
    def _get_numeric_stats(df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include='number').columns
        return {
            "mean": df[numeric_cols].mean().to_dict(),
            "median": df[numeric_cols].median().to_dict(),
            "std": df[numeric_cols].std().to_dict(),
            "min": df[numeric_cols].min().to_dict(),
            "max": df[numeric_cols].max().to_dict()
        }

    @staticmethod
    def _get_categorical_stats(df: pd.DataFrame) -> Dict[str, Any]:
        categorical_cols = df.select_dtypes(include='object').columns
        return {
            col: {
                "unique_count": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict()
            } for col in categorical_cols
        }

    @staticmethod
    def handle_missing_values(
            df: pd.DataFrame,
            strategy: str = 'mean',
            fill_value: Optional[Union[int, float, str]] = None,
            columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        处理缺失值
        :param df: 输入DataFrame
        :param strategy: 处理策略 ('mean', 'median', 'most_frequent', 'constant', 'drop')
        :param fill_value: 当strategy为'constant'时使用的填充值
        :param columns: 要处理的列列表，None表示处理所有列
        :return: 处理后的DataFrame
        """
        df = df.copy()
        if columns is None:
            columns = df.columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            # Skip if no missing values in this column
            if not df[col].isna().any():
                continue

            if strategy == 'mean':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    # For non-numeric columns, fall back to most_frequent
                    df[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # For non-numeric columns, fall back to most_frequent
                    df[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'most_frequent':
                df[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                if fill_value is not None:
                    df[col] = df[col].fillna(fill_value)
                else:
                    raise ValueError("fill_value must be specified for constant strategy")
            elif strategy == 'drop':
                df = df.dropna(subset=[col])
            else:
                raise ValueError(f"Unsupported strategy '{strategy}'")

        return {
            "processed_df": df,
        }

    @staticmethod
    def apply_feature_scaling(
            df: pd.DataFrame,
            method: str = 'standard',
            columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        应用特征缩放
        :param df: 输入DataFrame
        :param method: 缩放方法 ('standard', 'minmax', 'robust')
        :param columns: 要缩放的列列表，None表示处理所有数值列
        :return: 处理后的DataFrame
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include='number').columns.tolist()

        if not columns:
            return {
                "processed_df": df,
            }

        # 验证列存在
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in dataset: {', '.join(missing_columns)}")

        # 验证列类型
        non_numeric_columns = []
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_columns.append(col)

        if non_numeric_columns:
            raise ValueError(f"Non-numeric columns cannot be scaled: {', '.join(non_numeric_columns)}")

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

        df[columns] = scaler.fit_transform(df[columns])

        return {
            "processed_df": df,
        }

    @staticmethod
    def encode_categorical(
            df: pd.DataFrame,
            method: str = 'onehot',
            columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        编码分类变量
        :param df: 输入DataFrame
        :param method: 编码方法 ('onehot', 'label')
        :param columns: 要编码的列列表，None表示处理所有分类列
        :return: 处理后的DataFrame
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include='object').columns.tolist()

        if not columns:
            return {
                "processed_df": df,
            }
        print(columns, method)

        if method == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_data = encoder.fit_transform(df[columns])
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=encoder.get_feature_names_out(columns),
                index=df.index
            )
            df = pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)
        elif method == 'label':
            for col in columns:
                df[col] = df[col].astype('category').cat.codes
        else:
            raise ValueError(f"Unsupported encoding method: {method}")
        return {
            "processed_df": df,
        }

    @staticmethod
    def apply_pca(
            df: pd.DataFrame,
            n_components: int,
            columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        应用主成分分析（PCA）
        :param df: 输入DataFrame
        :param n_components: 保留的主成分数量
        :param columns: 要处理的列列表，None表示处理所有数值列
        :return: 处理后的DataFrame
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include='number').columns.tolist()

        if not columns:
            return {
                "processed_df": df,
            }

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(df[columns])
        pca_columns = [f'PC{i + 1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_columns, index=df.index)
        df = pd.concat([df.drop(columns, axis=1), pca_df], axis=1)
        return {
            "processed_df": df,
        }

    @staticmethod
    def handle_outliers(
            df: pd.DataFrame,
            method: str = 'zscore',
            threshold: float = 3,
            columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        处理异常值并返回处理详情
        :param df: 输入DataFrame
        :param method: 处理方法 ('zscore' 或 'iqr')
        :param threshold: 阈值
        :param columns: 要处理的列列表
        :return: 包含处理详情和结果DataFrame的字典
        """
        original_df = df.copy()
        original_count = len(df)

        # 确定要处理的列
        if columns is None:
            columns = df.select_dtypes(include='number').columns.tolist()

        # 过滤掉非数值列
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        columns = [col for col in columns if col in numeric_cols]

        if not columns:
            return {
                "processed_df": df,
                "removed_rows": 0,
                "method": method,
                "threshold": threshold,
                "message": "没有有效的数值列需要处理"
            }

        # 处理前记录异常值信息
        outlier_counts = {}
        for col in columns:
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col]))
                outlier_counts[col] = (z_scores >= threshold).sum()
            elif method == 'iqr':
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outlier_counts[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        # 执行异常值处理
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(df[columns]))
            df = df[(z_scores < threshold).all(axis=1)]
        elif method == 'iqr':
            mask = pd.Series(True, index=df.index)
            for col in columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                mask = mask & col_mask
            df = df[mask]
        else:
            raise ValueError(f"不支持的异常值处理方法: {method}")

        # 计算移除的行数
        removed_rows = original_count - len(df)

        # 准备返回结果
        return {
            "processed_df": df,
            "removed_rows": removed_rows,
            "method": method,
            "threshold": threshold,
            "columns": columns,
            "outlier_counts": outlier_counts,
            "message": f"移除了 {removed_rows} 行异常值"
        }

    @staticmethod
    def select_features(
            df: pd.DataFrame,
            columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        选择特征列
        :param df: 输入DataFrame
        :param columns: 要保留的列列表，None表示保留所有列
        :return: 处理后的DataFrame
        """
        df = df.copy()
        if columns is None:
            return {
                "processed_df": df,
            }
        valid_columns = [col for col in columns if col in df.columns]
        if not valid_columns:
            raise ValueError("No valid columns provided for feature selection")
        return {
            "processed_df": df[valid_columns],
        }