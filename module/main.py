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
from module.utils.data_module import fMRIDataModule, FA3DDataModule
from module.utils.parser import str2bool
from module.pl_classifier import LitClassifier,LitFA3DClassifier

def cli_main():

    # ------------ args -------------
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=1234, type=int, help="random seeds. recommend aligning this argument with data split number to control randomness")
    parser.add_argument("--dataset_name", type=str, choices=["Triple","FA3D"], default="FA3D")
    parser.add_argument("--downstream_task", type=str, default="tri_class", choices=["tri_class"],help="downstream task")
    parser.add_argument("--downstream_task_type", type=str, default="classification", choices=["classification"], help="select either classification or regression according to your downstream task")
    parser.add_argument("--classifier_module", default="default", type=str, help="A name of lightning classifier module (outdated argument)")
    parser.add_argument("--loggername", default="tensorboard", type=str, help="A name of logger")
    parser.add_argument("--project_name", default="default", type=str, help="A name of project (Neptune)")
    parser.add_argument("--resume_ckpt_path", type=str, help="A path to previous checkpoint. Use when you want to continue the training from the previous checkpoints")
    parser.add_argument("--load_model_path", type=str, help="A path to the pre-trained model weight file (.pth)")
    parser.add_argument("--test_only", default="False", action='store_true', help="specify when you want to test the checkpoints (model weights)")
    parser.add_argument("--test_ckpt_path",type=str, help="A path to the previous checkpoint that intends to evaluate (--test_only should be True)")# default="E:/PYProject/fMRI_DTI_Pretrained/models/project/output/default/last-v1.ckpt",
    parser.add_argument("--freeze_feature_extractor", action='store_true', help="Whether to freeze the feature extractor (for evaluating the pre-trained weight)")


    temp_args, _ = parser.parse_known_args()

    # 根据数据集选择不同的模型和数据模块
    if temp_args.dataset_name == "FA3D":
        Classifier = LitFA3DClassifier  # 3D FA分类器
        Dataset = FA3DDataModule  # 3D FA数据模块
    else:
        Classifier = LitClassifier
        Dataset = fMRIDataModule

    
    # add two additional arguments
    parser = Classifier.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()  # This command blocks the help message of Trainer class.
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    #override parameters
    max_epochs = args.max_epochs
    num_nodes = args.num_nodes
    devices = args.devices
    project_name = args.project_name
    image_path = args.image_path

    if temp_args.resume_ckpt_path is not None:
        # resume previous experiment
        from module.utils.neptune_utils import get_prev_args
        args = get_prev_args(args.resume_ckpt_path, args)
        exp_id = args.id
        # override max_epochs if you hope to prolong the training
        args.project_name = project_name
        args.max_epochs = max_epochs
        args.num_nodes = num_nodes
        args.devices = devices
        args.image_path = image_path       
    else:
        exp_id = None

    setattr(args, "default_root_dir", f"output/{args.project_name}/{args.dataset_name}")

    # ------------ data -------------
    data_module = Dataset(**vars(args))
    pl.seed_everything(args.seed)

    # ------------ logger -------------
    if args.loggername == "tensorboard":
        # logger = True  # tensor board is a default logger of Trainer class
        dirpath = args.default_root_dir
        logger = TensorBoardLogger(dirpath)
    elif args.loggername == "neptune":
        API_KEY = os.environ.get("NEPTUNE_API_TOKEN")
        # project_name should be "WORKSPACE_NAME/PROJECT_NAME"
        run = neptune.init(api_token=API_KEY, project=args.project_name, capture_stdout=False, capture_stderr=False, capture_hardware_metrics=False, run=exp_id)
        
        if exp_id == None:
            setattr(args, "id", run.fetch()['sys']['id'])

        logger = NeptuneLogger(run=run, log_model_checkpoints=False)
        dirpath = os.path.join(args.default_root_dir, logger.version)
    else:
        raise Exception("Wrong logger name.")

    # ------------ callbacks -------------
    # callback for pretraining task
    if args.pretraining:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_loss",
            filename="checkpt-{epoch:02d}-{valid_loss:.2f}",
            save_last=True,
            mode="min",
        )
    elif args.downstream_task_type == "classification":
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_acc",
            filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
            save_last=True,
            mode="max",
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    # ------------ trainer -------------
    if args.grad_clip:
        print('using gradient clipping')
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
            track_grad_norm=-1,
            accelerator="gpu",
            devices=1,
        )
    else:
        print('not using gradient clipping')
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

    # ------------ model -------------
    model = Classifier(data_module = data_module, **vars(args))

    if args.load_model_path is not None:
        print(f'loading model from {args.load_model_path}')
        path = args.load_model_path
        ckpt = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if 'model.' in k: #transformer-related layers
                new_state_dict[k.removeprefix("model.")] = v
        model.model.load_state_dict(new_state_dict)

    if args.freeze_feature_extractor:
        # layers are frozen by using eval()
        model.model.eval()
        # freeze params
        for name, param in model.model.named_parameters():
            if 'output_head' not in name: # unfreeze only output head
                param.requires_grad = False
                print(f'freezing layer {name}')

    # ------------ run -------------
    if args.test_only == True:
        trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path) # dataloaders=data_module
    else:
        if args.resume_ckpt_path is None:
            # New run
            trainer.fit(model, datamodule=data_module)
        else:
            # Resume existing run
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

        trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    cli_main()
