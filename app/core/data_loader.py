import pandas as pd


class DataLoader:
    """数据加载器，支持多种格式文件加载"""
    def __init__(self):
        super().__init__()

    def load_file(self, file_path):
        """加载文件方法（支持CSV、Excel和TXT）"""
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(file_path)
            elif file_path.endswith('.txt'):
                # 尝试自动检测分隔符加载TXT文件
                data = self._load_txt_with_auto_delimiter(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path.split('.')[-1]}")

            if data.empty:
                raise ValueError("加载的数据为空")

            return data

        except Exception as e:
            # 转换异常为更友好的错误信息
            if "No such file or directory" in str(e):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            elif "Password" in str(e):
                raise PermissionError("文件可能受密码保护")
            else:
                raise Exception(f"加载文件失败: {str(e)}")

    def _load_txt_with_auto_delimiter(self, file_path):
        """尝试用常见分隔符加载TXT文件"""
        # 常见分隔符列表（按优先级尝试）
        delimiters = [',', '\t', ';', '|', ' ']

        for delimiter in delimiters:
            try:
                data = pd.read_csv(file_path, delimiter=delimiter)
                # 验证是否成功读取了多列数据
                if len(data.columns) > 1:
                    return data
            except:
                continue

        # 如果常见分隔符都失败，尝试无分隔符读取
        try:
            return pd.read_csv(file_path, delimiter=None)
        except Exception as e:
            raise ValueError(f"无法确定TXT文件的分隔符: {str(e)}")

    def load_csv(self, file_path):
        """加载CSV文件"""
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            raise Exception(f"加载CSV失败: {str(e)}")

    def load_excel(self, file_path):
        """加载Excel文件"""
        try:
            data = pd.read_excel(file_path)
            return data
        except Exception as e:
            raise Exception(f"加载Excel失败: {str(e)}")

    def load_txt(self, file_path, delimiter=None):
        """加载TXT文件（可指定分隔符）"""
        try:
            if delimiter is None:
                data = self._load_txt_with_auto_delimiter(file_path)
            else:
                data = pd.read_csv(file_path, delimiter=delimiter)

            return data
        except Exception as e:
            raise Exception(f"加载TXT文件失败: {str(e)}")

dataloader = DataLoader()