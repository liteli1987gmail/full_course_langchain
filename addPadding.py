import os
import subprocess

# 指定 cn_docs 文件夹的路径
folder_path = './menu.md'

def batchAdd(folder_path=folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # 构建文件的完整路径
            file_path = os.path.join(root, file_name)
            print(file_path)
            file_path = file_path.replace(' ','')  # 删除空字符串
            # 构建要执行的命令
            command = f'npx md-padding -i {file_path}'
        
            # 执行命令
            subprocess.run(command, shell=True)

if not folder_path.endswith('md'):
    # print(folder_path)
    batchAdd(folder_path)
else:
    file_path = folder_path
    command = f'npx md-padding -i {file_path}'
    subprocess.run(command, shell=True)            
