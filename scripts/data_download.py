import kagglehub
import shutil
import os

# 下载到 kagglehub 默认缓存目录
path = kagglehub.competition_download("imaterialist-fashion-2019-FGVC6")

print("默认下载位置：", path)

# 目标目录
target_path = r"E:\Kaggle\imaterialist-fashion-2019-FGVC6"

# 如果目标目录已存在，可以先删除，避免冲突
if os.path.exists(target_path):
    shutil.rmtree(target_path)

# 复制到 E 盘
shutil.copytree(path, target_path)

print("已复制到：", target_path)
