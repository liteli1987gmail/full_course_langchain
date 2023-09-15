import os
import shutil

def clean_name(name):
    invalid_chars = '<>:"/\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, "")
    name = name.replace("\t", " ").strip()
    return name

def create_files_from_md(md_file):
    should_exist = set()
    with open(md_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = clean_name(line.strip())
        if line.startswith("第"):
            os.makedirs(line, exist_ok=True)
            current_dir = line
            should_exist.add(line)
        elif '.' in line and line.count('.') < 2:
            sub_dir = os.path.join(current_dir, line)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            current_sub_dir = sub_dir
            should_exist.add(sub_dir)
        elif '.' in line and line.count('.') >= 2:
            md_file_name = line + ".md"
            md_file_path = os.path.join(current_sub_dir, md_file_name)
            should_exist.add(md_file_path)
            if not os.path.exists(md_file_path):
                with open(md_file_path, 'w', encoding='utf-8') as md_file:
                    md_file.write("# " + line)

# create_files_from_md("menu.md")

import os

def list_files(startpath, output_file):
    with open(output_file, 'w',encoding='utf-8') as f:
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            f.write('{}{}/\n'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for name in files:
                f.write('{}{}\n'.format(subindent, name))

# list_files('./','./output.txt')

import os
import subprocess

# 指定Markdown文件夹和输出的Word文件夹
md_folder = 'pages/第2'
doc_folder = 'DOC'

# 指定Word模板
template = './custom-reference.docx'

# 创建一个函数，用于合并Markdown文件
def merge_files(file_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in file_list:
            print('Reading file:', fname)  # 打印正在读取的文件名
            with open(fname, 'r', encoding='utf-8') as infile:
                file_content = infile.read()
            print('Writing content to:', output_file)  # 打印正在写入的文件名
            outfile.write(file_content)
            outfile.write("\n")  # 在文件之间添加一个空行，以保持文件的分隔清晰

def process_folder(current_folder, output_folder):
    # 如果当前文件夹的名字以“第”开头，那么我们将所有的Markdown文件合并为一个大的Markdown文件
    if os.path.basename(current_folder).startswith("第"):
        md_files = []
        for root, dirs, files in os.walk(current_folder):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
        
        if md_files:
            # 合并这些Markdown文件为一个大的Markdown文件
            merged_md_file = os.path.join(current_folder, 'merged.md')
            merge_files(md_files, merged_md_file)
            # 然后将这个大的Markdown文件转换为一个Word文档
            output_path = os.path.join(output_folder, os.path.basename(current_folder) + '.docx')
            subprocess.run(['pandoc', '--reference-doc', template, '-s', merged_md_file, '-o', output_path])
    else:
        # 如果当前文件夹的名字不以“第”开头，那么我们遍历这个文件夹下的所有子文件夹
        for dir in os.listdir(current_folder):
            dir_path = os.path.join(current_folder, dir)
            if os.path.isdir(dir_path):
                # 在输出文件夹中创建相应的子文件夹
                new_output_folder = os.path.join(output_folder, dir)
                os.makedirs(new_output_folder, exist_ok=True)
                # 递归地处理这个子文件夹
                process_folder(dir_path, new_output_folder)

process_folder(md_folder,doc_folder)

# import os

# def replace_spaces_with_underscore(directory):
#     # 先遍历所有的文件并重命名
#     for foldername, subfolders, filenames in os.walk(directory):
#         for filename in filenames:
#             if " " in filename:
#                 new_filename = filename.replace(" ", "_")
#                 old_file_path = os.path.join(foldername, filename)
#                 new_file_path = os.path.join(foldername, new_filename)
#                 os.rename(old_file_path, new_file_path)

#     # 再遍历所有的文件夹并重命名
#     for foldername, subfolders, filenames in os.walk(directory):
#         if " " in foldername:
#             new_foldername = foldername.replace(" ", "_")
#             os.rename(foldername, new_foldername)


# # 使用函数
# # 注意：请替换 'your_directory' 为你需要处理的具体目录路径
# # replace_spaces_with_underscore('./pages/')






