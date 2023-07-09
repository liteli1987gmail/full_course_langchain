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
    
    # for root, dirs, files in os.walk(".", topdown=False):
    #     # Exclude .git folder
    #     dirs[:] = [d for d in dirs if d not in ['.git']]
    #     for name in files:
    #         file_path = os.path.join(root, name)
    #         if file_path not in should_exist:
    #             os.remove(file_path)
    #     for name in dirs:
    #         dir_path = os.path.join(root, name)
    #         if dir_path not in should_exist:
    #             shutil.rmtree(dir_path)

create_files_from_md("menu.md")
