import os
import shutil
import re

# 源目录根路径
source_root = r"G:\ADNI_CN"

# 目标目录
funimg_dir = r"G:\ADNI_CN\FunImg"
t1img_dir = r"G:\ADNI_CN\T1Img"

# 确保目标目录存在
os.makedirs(funimg_dir, exist_ok=True)
os.makedirs(t1img_dir, exist_ok=True)

# 遍历源目录中的所有sub-*文件夹
for sub_dir in os.listdir(source_root):
    if not sub_dir.startswith("sub-") or not os.path.isdir(os.path.join(source_root, sub_dir)):
        continue

    # 提取数字ID (如001)
    sub_id = re.search(r"sub-(\d+)", sub_dir)
    if not sub_id:
        continue

    sub_num = sub_id.group(1)
    new_sub_name = f"sub_{sub_num}"

    # 处理func文件夹
    func_source = os.path.join(source_root, sub_dir, "func")
    if os.path.exists(func_source):
        func_dest = os.path.join(funimg_dir, new_sub_name)
        os.makedirs(func_dest, exist_ok=True)

        # 复制所有文件
        for item in os.listdir(func_source):
            src_path = os.path.join(func_source, item)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, func_dest)
                print(f"Copied: {src_path} -> {func_dest}")

    # 处理anat文件夹
    anat_source = os.path.join(source_root, sub_dir, "anat")
    if os.path.exists(anat_source):
        anat_dest = os.path.join(t1img_dir, new_sub_name)
        os.makedirs(anat_dest, exist_ok=True)

        # 复制所有文件
        for item in os.listdir(anat_source):
            src_path = os.path.join(anat_source, item)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, anat_dest)
                print(f"Copied: {src_path} -> {anat_dest}")

print("\n文件复制完成！")