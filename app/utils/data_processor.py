import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


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
    ) -> pd.DataFrame:
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
                    df[col] = df[col].fillna(df[col].mean())  # Fixed inplace warning
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

        return df

    @staticmethod
    def apply_feature_scaling(
            df: pd.DataFrame,
            method: str = 'standard',
            columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        应用特征缩放
        :param df: 输入DataFrame
        :param method: 缩放方法 ('standard', 'minmax')
        :param columns: 要缩放的列列表，None表示处理所有数值列
        :return: 处理后的DataFrame
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include='number').columns.tolist()

        if not columns:
            return df

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

        df[columns] = scaler.fit_transform(df[columns])
        return df

    @staticmethod
    def encode_categorical(
            df: pd.DataFrame,
            method: str = 'onehot',
            columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
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
            return df

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

        return df