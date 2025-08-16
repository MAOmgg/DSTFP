import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# from module import LitClassifier
import neptune as neptune
from module.utils.data_module import fMRIDataModule, FA3DDataModule,MultimodalDataModule
from module.utils.parser import str2bool
from module.pl_classifier import LitClassifier,LitFA3DClassifier,LitMultimodalClassifier


def cli_main():
    # ------------ args -------------
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=1234, type=int, help="随机种子")
    parser.add_argument("--dataset_name", type=str, choices=["Multimodal", "Triple", "FA3D"], default="Multimodal")
    parser.add_argument("--downstream_task", type=str, default="tri_class", choices=["tri_class"])
    parser.add_argument("--downstream_task_type", type=str, default="classification", choices=["classification"])
    parser.add_argument("--loggername", default="tensorboard", type=str, help="日志记录器名称")
    parser.add_argument("--project_name", default="multimodal_pretrain", type=str, help="项目名称")
    parser.add_argument("--resume_ckpt_path", type=str, help="恢复训练检查点路径")
    parser.add_argument("--load_model_path", type=str, help="预训练模型权重路径")
    parser.add_argument("--test_only", action='store_true', help="仅测试模式")
    parser.add_argument("--test_ckpt_path", type=str, help="测试检查点路径")
    parser.add_argument("--freeze_feature_extractor", action='store_true', help="冻结特征提取器")
    parser.add_argument("--fmri_root", type=str, default="E:\Python\File\Combined_fmri_dti_pretrained\Data", help="fMRI数据根目录")
    parser.add_argument("--fa_root", type=str, default="E:\Python\File\Combined_fmri_dti_pretrained\Data", help="FA数据根目录")

    temp_args, _ = parser.parse_known_args()

    # 根据数据集选择不同的模型和数据模块
    if temp_args.dataset_name == "Multimodal":
        Classifier = LitMultimodalClassifier
        Dataset = MultimodalDataModule
    elif temp_args.dataset_name == "FA3D":
        Classifier = LitFA3DClassifier
        Dataset = FA3DDataModule
    else:
        Classifier = LitClassifier
        Dataset = fMRIDataModule

    # 添加特定于模型和数据的参数
    parser = Classifier.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # 设置默认根目录
    setattr(args, "default_root_dir", f"output/{args.project_name}/{args.dataset_name}")

    # 创建数据模块
    if args.dataset_name == "Multimodal":
        data_module = MultimodalDataModule(**vars(args))
    else:
        data_module = Dataset(**vars(args))

    pl.seed_everything(args.seed)

    # ------------ 日志记录器 -------------
    if args.loggername == "tensorboard":
        dirpath = args.default_root_dir
        logger = TensorBoardLogger(dirpath)
    elif args.loggername == "neptune":
        API_KEY = os.environ.get("NEPTUNE_API_TOKEN")
        run = neptune.init(api_token=API_KEY, project=args.project_name,
                           capture_stdout=False, capture_stderr=False,
                           capture_hardware_metrics=False)
        logger = NeptuneLogger(run=run, log_model_checkpoints=False)
        dirpath = os.path.join(args.default_root_dir, logger.version)
    else:
        raise Exception("错误的日志记录器名称。")

    # ------------ 回调函数 -------------
    if args.downstream_task_type == "classification":
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_acc",
            filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
            save_last=True,
            mode="max",
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    # ------------ 训练器 -------------
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        check_val_every_n_epoch=1,  # 每个 epoch 进行一次验证
        val_check_interval=1.0,  # 每个 epoch 结束时进行验证
        callbacks=callbacks,
        accelerator="gpu",  # 指定使用 GPU
        devices=1,  # 指定使用的 GPU 数量，例如 1 个 GPU
        max_epochs=100,  # 设置最大 epoch 数
        log_every_n_steps=1,  # 每5个批次记录一次日志
    )

    # ------------ 模型 -------------
    model = Classifier(data_module=data_module, **vars(args))

    # 加载预训练权重
    if args.load_model_path is not None:
        print(f'从 {args.load_model_path} 加载模型')
        ckpt = torch.load(args.load_model_path)
        model.load_state_dict(ckpt['state_dict'])

    # 冻结特征提取器
    if args.freeze_feature_extractor:
        for param in model.fmri_model.parameters():
            param.requires_grad = False
        for param in model.fa_model.parameters():
            param.requires_grad = False
        print('冻结特征提取器')

    # ------------ 运行 -------------
    if args.test_only:
        trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path)
    else:
        if args.resume_ckpt_path is None:
            # 新训练
            trainer.fit(model, datamodule=data_module)
        else:
            # 恢复训练
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

        trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    cli_main()