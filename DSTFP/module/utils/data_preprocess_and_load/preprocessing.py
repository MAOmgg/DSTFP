import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复加载OpenMP
os.environ['OMP_NUM_THREADS'] = '1'  # 限制线程数为1
from monai.transforms import LoadImage
import torch
import numpy as np
import time
from multiprocessing import Process
import sys
import nibabel as nib  # 添加nibabel直接加载


def read_data(filename, load_root, save_root, subj_name, count, scaling_method=None, fill_zeroback=False):
    try:
        print(f"\n{'=' * 50}\n处理 #{count}: {filename}", flush=True)
        path = os.path.join(load_root, filename)
        print(f"完整路径: {path}")

        # 检查文件是否存在
        if not os.path.exists(path):
            print(f"文件不存在: {path}")
            return None

        # 检查文件大小
        file_size = os.path.getsize(path)
        print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")

        # 更健壮的NIfTI文件加载方法
        try:
            # 方法1: 使用monai的LoadImage
            data = LoadImage()(path)
            print(f"MONAI加载成功: {filename}, 形状: {data.shape}, 类型: {type(data)}")
        except Exception as e:
            print(f"MONAI加载失败，尝试nibabel: {str(e)}")
            try:
                # 方法2: 使用nibabel直接加载
                img = nib.load(path)
                data = img.get_fdata()
                print(f"NiBabel加载成功: {filename}, 形状: {data.shape}, 类型: {type(data)}")
            except Exception as e:
                print(f"NiBabel加载失败: {str(e)}")
                return None

        # 确保数据是torch张量
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(np.asarray(data))
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
    print(f"原始数据形状: {data.shape}")
    if data.ndim != 4:
        print(f"警告: 数据维度为{data.ndim}，期望4D数据(x,y,z,time)")
        # 尝试修复3D数据
        if data.ndim == 3:
            data = data.unsqueeze(-1)  # 添加时间维度
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

        if scaling_method == 'z-norm' and non_background.numel() > 0:
            global_mean = non_background.mean()
            global_std = non_background.std()
            # 避免除零错误
            if global_std == 0:
                global_std = 1e-6
            print(f"Z标准化: 均值={global_mean:.2f}, 标准差={global_std:.2f}")
            data_temp = (data - global_mean) / global_std
        elif scaling_method == 'minmax' and non_background.numel() > 0:
            data_min = non_background.min()
            data_max = non_background.max()
            # 避免除零错误
            if data_max - data_min == 0:
                data_max = data_min + 1e-6
            print(f"MinMax标准化: 最小值={data_min:.2f}, 最大值={data_max:.2f}")
            data_temp = (data - data_min) / (data_max - data_min)
        else:
            print("使用原始数据，未标准化")
            data_temp = data
    except Exception as e:
        print(f"标准化错误: {str(e)}")
        return None

    # 背景处理
    try:
        if non_background.numel() > 0:
            min_val = data_temp[~background].min()
        else:
            min_val = 0

        data_global = torch.empty_like(data_temp)
        data_global[background] = min_val if not fill_zeroback else 0
        if non_background.numel() > 0:
            data_global[~background] = data_temp[~background]
        print(f"背景值设置为: {data_global[background].mean():.2f}")
    except Exception as e:
        print(f"背景处理错误: {str(e)}")
        return None

    # 保存处理结果
    try:
        data_global = data_global.type(torch.float16)
        # 按时间维度分割
        data_global_split = torch.unbind(data_global, dim=-1)
        print(f"将分割为 {len(data_global_split)} 个时间点")

        saved_count = 0
        for i, TR in enumerate(data_global_split):
            save_path = os.path.join(save_dir, f"frame_{i}.pt")
            # 移除单维度并保存
            torch.save(TR.squeeze(-1).clone(), save_path)
            saved_count += 1

            # 只打印前几个文件的信息
            if i < 3:
                print(f"保存: {save_path} - 形状: {TR.squeeze(-1).shape}")

        print(f"成功保存 {saved_count} 个TR文件到 {save_dir}")
        return True

    except Exception as e:
        print(f"保存错误: {str(e)}")
        return None


def main():
    # 配置参数
    load_root = r'E:\PYProject\fMRI_DTI_Pretrained\Data\OASIS3'
    save_root = r'E:\PYProject\fMRI_DTI_Pretrained\Data\OASIS3_MNI_to_TRs_minmax'
    scaling_method = 'z-norm'
    fill_zeroback = False

    print("\n" + "=" * 50)
    print(f"fMRI 预处理流水线")
    print(f"输入目录: {load_root}")
    print(f"输出目录: {save_root}")
    print(f"标准化方法: {scaling_method}")
    print(f"背景填充: {'0' if fill_zeroback else '最小值'}")
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

    # 获取文件列表 - 支持 .nii 和 .nii.gz
    filenames = [f for f in os.listdir(load_root)
                 if f.endswith('.nii') or f.endswith('.nii.gz')]

    if not filenames:
        print(f"错误: 在 {load_root} 中没有找到NIfTI文件")
        return

    print(f"找到 {len(filenames)} 个NIfTI文件")
    print(f"前5个文件: {filenames[:5]}")

    # 处理文件
    processed_count = 0
    success_count = 0
    skip_count = 0
    error_count = 0

    for count, filename in enumerate(sorted(filenames), 1):
        if filename.endswith('.nii.gz'):
            subj_name = filename[:-7]  # 移除 '.nii.gz'
        else:  # .nii 文件
            subj_name = filename[:-4]  # 移除 '.nii'

        subject_dir = os.path.join(img_dir, subj_name)
        subject_exists = os.path.exists(subject_dir)

        # 检查是否已完成处理
        if subject_exists:
            existing_files = [f for f in os.listdir(subject_dir) if f.endswith('.pt')]
            if existing_files:
                print(f"跳过已处理的主题: {subj_name} (已有 {len(existing_files)} 个TR文件)")
                skip_count += 1
                continue

        try:
            print(f"\n启动处理 #{count}: {filename}")
            start_time = time.time()

            result = read_data(
                filename,
                load_root,
                img_dir,
                subj_name,
                count,
                scaling_method=scaling_method,
                fill_zeroback=fill_zeroback
            )

            if result:
                success_count += 1
                print(f"成功处理 #{count} 用时: {time.time() - start_time:.1f}秒")
            else:
                error_count += 1
                print(f"处理失败 #{count}")

            processed_count += 1

        except Exception as e:
            error_count += 1
            print(f'处理 {filename} 时出错: {str(e)}')
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"处理完成! 总文件数: {len(filenames)}")
    print(f"已处理: {processed_count}, 成功: {success_count}, 失败: {error_count}, 跳过: {skip_count}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f'\n总共耗时: {duration:.1f} 分钟')