import os
import shutil
import re  # 导入正则表达式模块

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复加载OpenMP
os.environ['OMP_NUM_THREADS'] = '1'  # 限制线程数为1
import torch
import numpy as np
import time
import nibabel as nib


def process_fa_data(filename, load_root, save_root, subj_name, count, scaling_method=None):
    """
    专门处理FA图像数据的函数
    FA图像通常是3D张量，没有时间维度
    """
    try:
        print(f"\n{'=' * 50}\n处理FA数据 #{count}: {filename}", flush=True)
        path = os.path.join(load_root, filename)
        print(f"完整路径: {path}")

        # 检查文件是否存在
        if not os.path.exists(path):
            print(f"文件不存在: {path}")
            return None

        # 检查文件大小
        file_size = os.path.getsize(path)
        print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")

        # 加载FA图像
        try:
            # 使用nibabel直接加载FA图像
            img = nib.load(path)
            data = img.get_fdata()
            print(f"NiBabel加载成功: {filename}, 形状: {data.shape}, 类型: {type(data)}")
        except Exception as e:
            print(f"加载失败: {str(e)}")
            return None

        # 转换为torch张量
        data = torch.from_numpy(data.astype(np.float32))
        print(f"转换为torch.Tensor: {data.shape}")

    except Exception as e:
        print(f"加载失败: {filename}, 错误: {str(e)}")
        return None

    # 创建保存目录
    save_dir = os.path.join(save_root, subj_name)
    print(f"保存目录: {save_dir}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建目录: {save_dir}")

    # 检查数据维度
    print(f"FA数据形状: {data.shape}")
    if data.ndim != 3:
        print(f"警告: FA数据维度为{data.ndim}，期望3D数据(x,y,z)")
        # 尝试修复维度问题
        if data.ndim == 4:
            print("检测到4D数据，取第一帧作为FA图")
            data = data[..., 0]
            print(f"修复后形状: {data.shape}")
        else:
            print("不支持的维度，跳过处理")
            return None

    # 背景检测
    background = data == 0
    print(f"背景体素数: {background.sum().item()}, 脑区体素数: {torch.sum(~background).item()}")

    # 标准化处理
    try:
        non_background = data[~background]
        brain_mask = ~background

        if scaling_method == 'z-norm' and non_background.numel() > 0:
            brain_mean = non_background.mean()
            brain_std = non_background.std()
            # 避免除零错误
            if brain_std == 0:
                brain_std = 1e-6
            print(f"Z标准化: 脑区均值={brain_mean:.4f}, 标准差={brain_std:.4f}")
            data[brain_mask] = (data[brain_mask] - brain_mean) / brain_std
        elif scaling_method == 'minmax' and non_background.numel() > 0:
            brain_min = non_background.min()
            brain_max = non_background.max()
            # 避免除零错误
            if brain_max - brain_min == 0:
                brain_max = brain_min + 1e-6
            print(f"MinMax标准化: 脑区最小值={brain_min:.4f}, 最大值={brain_max:.4f}")
            data[brain_mask] = (data[brain_mask] - brain_min) / (brain_max - brain_min)
        else:
            print("使用原始数据，未标准化")

        # 确保背景保持为0
        data[background] = 0

    except Exception as e:
        print(f"标准化错误: {str(e)}")
        return None

    # 保存处理结果
    try:
        save_path = os.path.join(save_dir, "fa.pt")
        # 保存为float16以节省空间
        torch.save(data.type(torch.float16).clone(), save_path)
        print(f"成功保存FA数据到: {save_path} - 形状: {data.shape}")

        # 额外保存为NIfTI文件用于检查
        nii_save_path = os.path.join(save_dir, "fa_processed.nii.gz")
        processed_img = nib.Nifti1Image(data.numpy().astype(np.float32), img.affine)
        nib.save(processed_img, nii_save_path)
        print(f"额外保存为NIfTI文件: {nii_save_path}")

        return True

    except Exception as e:
        print(f"保存错误: {str(e)}")
        return None


def extract_subject_id(filename):
    """
    从文件名中提取主题ID
    支持格式: sub-001_ses-01_run-01_dwi_dti_fa.nii.gz -> 返回 "001"
    """
    # 使用正则表达式匹配 sub- 后面的数字部分
    match = re.search(r'sub-(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)  # 返回数字部分

    # 尝试匹配其他可能的格式
    match = re.search(r'sub_(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r'sub(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def main():
    # 配置参数 - 针对FA图像
    load_root = r'E:\PYProject\fMRI_DTI_Pretrained\Data\Prodromal_FA'  # FA图像目录
    save_root = r'E:\PYProject\fMRI_DTI_Pretrained\Data\Prodromal_FA_processed'  # 处理后的FA保存目录
    scaling_method = 'z-norm'  # FA标准化方法: 'z-norm', 'minmax' 或 None

    print("\n" + "=" * 50)
    print(f"FA 图像预处理流水线")
    print(f"输入目录: {load_root}")
    print(f"输出目录: {save_root}")
    print(f"标准化方法: {scaling_method}")
    print("=" * 50 + "\n")

    # 检查输入路径
    if not os.path.exists(load_root):
        print(f"错误: 输入路径不存在 - {load_root}")
        return
    print(f"输入路径存在: {os.path.exists(load_root)}")

    # 创建输出目录
    img_dir = os.path.join(save_root, 'img')
    os.makedirs(img_dir, exist_ok=True)
    print(f"创建输出目录: {img_dir}")

    # 检查写入权限
    test_file = os.path.join(img_dir, 'test_permission.txt')
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("输出目录写入权限: 正常")
    except Exception as e:
        print(f"输出目录写入权限错误: {str(e)}")
        return

    # 获取文件列表 - FA图像通常为.nii或.nii.gz格式
    fa_filenames = [f for f in os.listdir(load_root)
                    if f.lower().endswith(('.nii', '.nii.gz', '.fa.nii', '.fa.nii.gz'))]

    if not fa_filenames:
        print(f"错误: 在 {load_root} 中没有找到FA图像文件")
        return

    print(f"找到 {len(fa_filenames)} 个FA图像文件")
    print(f"前5个文件: {fa_filenames[:5]}")

    # 处理FA文件
    processed_count = 0
    success_count = 0
    skip_count = 0
    error_count = 0
    subject_counter = 1  # 添加主题计数器

    # 创建主题ID到新名称的映射
    subject_id_to_new_name = {}

    for count, filename in enumerate(sorted(fa_filenames), 1):
        # 提取主题ID
        subject_id = extract_subject_id(filename)

        if subject_id is None:
            print(f"无法从文件名中提取主题ID: {filename}")
            error_count += 1
            continue

        # 原始主题名称 - 使用提取的数字ID
        orig_subj_name = f"sub-{subject_id}"

        # 检查是否已经处理过这个主题
        if subject_id in subject_id_to_new_name:
            new_subj_name = subject_id_to_new_name[subject_id]
            subject_dir = os.path.join(img_dir, new_subj_name)
            fa_file = os.path.join(subject_dir, "fa.pt")

            if os.path.exists(fa_file):
                print(f"跳过已处理的FA主题: {new_subj_name} (原始文件: {filename})")
                skip_count += 1
                continue

        # 创建主题目录（使用原始名称）
        subject_dir = os.path.join(img_dir, orig_subj_name)
        subject_exists = os.path.exists(subject_dir)

        try:
            print(f"\n启动FA处理 #{count}: {filename}")
            start_time = time.time()

            result = process_fa_data(
                filename,
                load_root,
                img_dir,
                orig_subj_name,  # 使用原始主题名称
                count,
                scaling_method=scaling_method
            )

            if result:
                success_count += 1
                print(f"成功处理FA #{count} 用时: {time.time() - start_time:.1f}秒")

                # 生成新主题名称
                new_subj_name = f"sub{subject_counter:03d}"

                # 重命名文件夹
                new_subject_dir = os.path.join(img_dir, new_subj_name)

                # 确保新文件夹名称不存在
                if os.path.exists(new_subject_dir):
                    print(f"警告: 目标文件夹已存在 {new_subject_dir}, 删除旧数据")
                    shutil.rmtree(new_subject_dir)

                # 重命名文件夹
                os.rename(subject_dir, new_subject_dir)
                print(f"已将文件夹 {orig_subj_name} 重命名为: {new_subj_name}")

                # 记录映射关系
                subject_id_to_new_name[subject_id] = new_subj_name

                subject_counter += 1  # 递增主题计数器
            else:
                error_count += 1
                print(f"FA处理失败 #{count}")

            processed_count += 1

        except Exception as e:
            error_count += 1
            print(f'处理 {filename} 时出错: {str(e)}')
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"FA处理完成! 总文件数: {len(fa_filenames)}")
    print(f"已处理: {processed_count}, 成功: {success_count}, 失败: {error_count}, 跳过: {skip_count}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f'\n总共耗时: {duration:.1f} 分钟')