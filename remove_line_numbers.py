import os
import re

def remove_line_numbers_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            new_line = re.sub(r'^\d+\|', '', line)
            f.write(new_line)

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                remove_line_numbers_from_file(file_path)
                print(f"已处理文件：{file_path}")

if __name__ == "__main__":
    # 将 '您的项目路径' 替换为实际的项目路径
    process_directory('/Users/minimax/PycharmProjects/LOLOracle')

