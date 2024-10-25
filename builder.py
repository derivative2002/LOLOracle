import os
import logging


class ProjectStructure:
    """项目目录结构生成器。"""

    def __init__(self, base_path='.'):
        """初始化。

        Args:
            base_path (str): 项目根目录路径。
        """
        self.base_path = base_path
        self.dirs = [
            'data/raw',
            'data/processed',
            'src/data',
            'src/models',
            'src/utils',
            'notebooks',
            'logs',
            'outputs/models',
            'outputs/predictions',
            'config',
            'tests'  # 新增tests目录
        ]
        self.files = [
            'README.md',
            'requirements.txt',
            'src/train.py',
            'src/predict.py',
            'config/config.yaml',
            'config/logging.conf',  # 新增日志配置文件
            'notebooks/EDA.ipynb',
            'src/data/__init__.py',
            'src/data/data_loader.py',
            'src/data/data_preprocessor.py',
            'src/data/preprocess.py',  # 新增预处理模块
            'src/data/postprocess.py',  # 新增后处理模块
            'src/models/__init__.py',
            'src/models/model.py',
            'src/utils/__init__.py',
            'src/utils/utils.py',
            'tests/__init__.py',  # 新增测试模块的__init__.py
            'tests/test_data.py',  # 新增数据模块的测试文件
            'tests/test_models.py',  # 新增模型模块的测试文件
            'tests/test_utils.py'  # 新增工具模块的测试文件
        ]
        # 配置日志
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    def create_directories(self):
        """创建目录结构。"""
        for dir_path in self.dirs:
            full_path = os.path.join(self.base_path, dir_path)
            os.makedirs(full_path, exist_ok=True)
            logging.info(f'目录已创建: {full_path}')

    def create_files(self):
        """创建文件。"""
        for file_path in self.files:
            full_path = os.path.join(self.base_path, file_path)
            dir_name = os.path.dirname(full_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                logging.info(f'目录已创建: {dir_name}')
            if not os.path.exists(full_path):
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write('')
                logging.info(f'文件已创建: {full_path}')

    def create_project(self):
        """创建项目结构。"""
        self.create_directories()
        self.create_files()
        logging.info('项目目录结构创建完成。')


if __name__ == '__main__':
    project = ProjectStructure()
    project.create_project()
