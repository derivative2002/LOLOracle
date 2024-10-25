#!/bin/bash

# 切换到项目根目录
cd /data/minimax-dialogue/users/jiaoyang/LOLOracle

# 激活虚拟环境，如果需要的话
source /data/minimax-dialogue/users/jiaoyang/miniconda3/bin/activate datacleaning

# 运行训练脚本
python src/train.py
