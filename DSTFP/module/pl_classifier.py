
import logging
from datetime import datetime, time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import os
import pickle
import scipy

import torchmetrics
import torchmetrics.classification
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryROC
from torchmetrics import  PearsonCorrCoef # Accuracy,
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve
import monai.transforms as monai_t

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .models.fusion_H_W import AxialAttentionTransformer, optimized_fusion
# from .models.fusion_strategy import AxialAttentionTransformer, attention_mechanism_fusion

from .models.load_model import load_model
from .models.swin3d_transformer import SwinTransformer3D
from .models.swin4d_transformer_ver7 import SwinTransformer4D
from .utils.metrics import Metrics
from .utils.parser import str2bool
from .utils.losses import NTXentLoss, global_local_temporal_contrastive
from .utils.lr_scheduler import WarmupCosineSchedule, CosineAnnealingWarmUpRestarts
from einops import rearrange
from pytorch_lightning import LightningModule
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import confusion_matrix, classification_report
logger = logging.getLogger(__name__)

class LitClassifier(pl.LightningModule):
    def __init__(self,data_module, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.save_hyperparameters(kwargs) # save hyperparameters except data_module (data_module cannot be pickled as a checkpoint)

        # you should define target_values at the Dataset classes
        target_values = data_module.train_dataset.target_values
        if self.hparams.label_scaling_method == 'standardization':
            scaler = StandardScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
        elif self.hparams.label_scaling_method == 'minmax':
            scaler = MinMaxScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
        self.scaler = scaler
        print(self.hparams.model)
        self.model = load_model(self.hparams.model, self.hparams)

        # Heads
        if not self.hparams.pretraining:
            if self.hparams.downstream_task == 'tri_class':
                    self.output_head = load_model("clf_mlp", self.hparams)

        elif self.hparams.use_contrastive:
            self.output_head = load_model("emb_mlp", self.hparams)
        else:
            raise NotImplementedError("output head should be defined")

        self.metric = Metrics()

        if self.hparams.adjust_thresh:
            self.threshold = 0
        self.train_losses = []
        self.train_epoch_losses = []
        self.train_step_losses = []
    def forward(self, x):
        return self.output_head(self.model(x))

    def augment(self, img):

        B, C, H, W, D, T = img.shape

        device = img.device
        img = rearrange(img, 'b c h w d t -> b t c h w d')

        rand_affine = monai_t.RandAffine(
            prob=1.0,
            # 0.175 rad = 10 degrees
            rotate_range=(0.175, 0.175, 0.175),
            scale_range = (0.1, 0.1, 0.1),
            mode = "bilinear",
            padding_mode = "border",
            device = device
        )
        rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
        rand_smooth = monai_t.RandGaussianSmooth(sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1)
        if self.hparams.augment_only_intensity:
            comp = monai_t.Compose([rand_noise, rand_smooth])
        else:
            comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth])

        for b in range(B):
            aug_seed = torch.randint(0, 10000000, (1,)).item()
            # set augmentation seed to be the same for all time steps
            for t in range(T):
                if self.hparams.augment_only_affine:
                    rand_affine.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = rand_affine(img[b, t, :, :, :, :])
                else:
                    comp.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = comp(img[b, t, :, :, :, :])

        img = rearrange(img, 'b t c h w d -> b c h w d t')

        return img

    def _compute_logits(self, batch, augment_during_training=None):
        fmri, subj, target_value, tr = batch.values()

        if augment_during_training:
            fmri = self.augment(fmri)

        feature = self.model(fmri)
        if self.hparams.downstream_task == 'tri_class':
            logits = self.output_head(feature)
            target = target_value.float().squeeze()
        # Classification task
        elif self.hparams.downstream_task == 'sex' or self.hparams.scalability_check:
            logits = self.output_head(feature).squeeze() #self.clf(feature).squeeze()
            target = target_value.float().squeeze()
        # Regression task
        elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression':
            # target_mean, target_std = self.determine_target_mean_std()
            logits = self.output_head(feature) # (batch,1) or # tuple((batch,1), (batch,1))
            unnormalized_target = target_value.float() # (batch,1)
            if self.hparams.label_scaling_method == 'standardization': # default
                target = (unnormalized_target - self.scaler.mean_[0]) / (self.scaler.scale_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                target = (unnormalized_target - self.scaler.data_min_[0]) / (self.scaler.data_max_[0] - self.scaler.data_min_[0])

        return subj, logits, target

    def _calculate_loss(self, batch, mode):
        if self.hparams.pretraining:
            fmri, subj, target_value, tr, sex = batch.values()

            cond1 = (self.hparams.in_chans == 1 and not self.hparams.with_voxel_norm)
            assert cond1, "Wrong combination of options"
            loss = 0

            if self.hparams.use_contrastive:
                assert self.hparams.contrastive_type != "none", "Contrastive type not specified"

                # B, C, H, W, D, T = image shape
                y, diff_y = fmri

                batch_size = y.shape[0]
                if (len(subj) != len(tuple(subj))) and mode == 'train':
                    print('Some sub-sequences in a batch came from the same subject!')
                criterion = NTXentLoss(device='cuda', batch_size=batch_size,
                                        temperature=self.hparams.temperature,
                                        use_cosine_similarity=True).cuda()
                criterion_ll = NTXentLoss(device='cuda', batch_size=2,
                                            temperature=self.hparams.temperature,
                                            use_cosine_similarity=True).cuda()

                # type 1: IC
                # type 2: LL
                # type 3: IC + LL
                if self.hparams.contrastive_type in [1, 3]:
                    out_global_1 = self.output_head(self.model(self.augment(y)),"g")
                    out_global_2 = self.output_head(self.model(self.augment(diff_y)),"g")
                    ic_loss = criterion(out_global_1, out_global_2)
                    loss += ic_loss

                if self.hparams.contrastive_type in [2, 3]:
                    out_local_1 = []
                    out_local_2 = []
                    out_local_swin1 = self.model(self.augment(y))
                    out_local_swin2 = self.model(self.augment(y))
                    out_local_1.append(self.output_head(out_local_swin1, "l"))
                    out_local_2.append(self.output_head(out_local_swin2, "l"))

                    out_local_swin1 = self.model(self.augment(diff_y))
                    out_local_swin2 = self.model(self.augment(diff_y))
                    out_local_1.append(self.output_head(out_local_swin1, "l"))
                    out_local_2.append(self.output_head(out_local_swin2, "l"))

                    ll_loss = 0
                    # loop over batch size
                    for i in range(out_local_1[0].shape[0]):
                        # out_local shape should be: BS, n_local_clips, D
                        ll_loss += criterion_ll(torch.stack(out_local_1, dim=1)[i],
                                                torch.stack(out_local_2, dim=1)[i])
                    loss += ll_loss

                result_dict = {
                    f"{mode}_loss": loss,
                }
        else:
            subj, logits, target = self._compute_logits(batch, augment_during_training = self.hparams.augment_during_training)
            if self.hparams.downstream_task_type == 'classification':
                if self.hparams.downstream_task == 'tri_class':
                    loss = F.cross_entropy(logits, target.long().squeeze())
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == target.long().squeeze()).float().mean()

                    result_dict = {
                        f"{mode}_loss": loss,
                        f"{mode}_acc": acc,
                    }


        self.log_dict(result_dict, prog_bar=True, sync_dist=False, add_dataloader_idx=False, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size) # batch_size = batch_size
        if mode == "train":
            self.train_losses.append(loss.item())

        return loss

    def on_fit_end(self):
        if not self.trainer.is_global_zero:
            return

        plt.figure(figsize=(10, 6))

        epochs = range(len(self.train_epoch_losses))

        plt.plot(epochs, self.train_epoch_losses, 'b-', label='Training Loss', linewidth=2)

        plt.title('Training Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        if len(self.train_epoch_losses) > 0:
            if len(self.train_epoch_losses) <= 20:
                plt.xticks(range(len(self.train_epoch_losses)))
            else:
                step = max(1, len(self.train_epoch_losses) // 10)
                plt.xticks(range(0, len(self.train_epoch_losses), step))

        os.makedirs("training_loss_plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"training_loss_plots/training_loss_per_epoch_{timestamp}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training loss curve saved to training_loss_plots/training_loss_per_epoch_{timestamp}.png")
    def _evaluate_metrics(self, subj_array, total_out, mode):
        # print('total_out.device',total_out.device)
        # (total iteration/world_size) numbers of samples are passed into _evaluate_metrics.
        subjects = np.unique(subj_array)

        subj_avg_logits = []
        subj_targets = []
        for subj in subjects:
            #print('total_out.shape:',total_out.shape) # total_out.shape: torch.Size([16, 2])
            subj_logits = total_out[subj_array == subj,0]
            subj_avg_logits.append(torch.mean(subj_logits).item())
            subj_targets.append(total_out[subj_array == subj,1][0].item())
        subj_avg_logits = torch.tensor(subj_avg_logits, device = total_out.device)
        subj_targets = torch.tensor(subj_targets, device = total_out.device)

        if self.hparams.downstream_task == 'tri_class':
            subj_avg_logits = []
            subj_targets = []

            for subj in subjects:
                subj_logits = total_out[subj_array == subj, :self.num_classes]
                subj_probs = torch.softmax(subj_logits, dim=1)
                subj_avg_probs = torch.mean(subj_probs, dim=0)
                subj_avg_logits.append(subj_avg_probs)
                subj_targets.append(total_out[subj_array == subj, self.num_classes][0].item())

            subj_avg_logits = torch.stack(subj_avg_logits)
            subj_targets = torch.tensor(subj_targets, device=total_out.device, dtype=torch.long)

            preds = torch.argmax(subj_avg_logits, dim=1)
            acc = (preds == subj_targets).float().mean()

            bal_acc = balanced_accuracy_score(
                subj_targets.cpu().numpy(),
                preds.cpu().numpy()
            )

            auroc = torchmetrics.classification.MulticlassAUROC(
                num_classes=self.num_classes
            ).to(total_out.device)
            auroc_score = auroc(subj_avg_logits, subj_targets)

            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_balacc", bal_acc, sync_dist=True)
            self.log(f"{mode}_AUROC", auroc_score, sync_dist=True)


    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        self.train_step_losses.append(loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if self.hparams.pretraining:
            if dataloader_idx == 0:
                self._calculate_loss(batch, mode="valid")
            else:
                self._calculate_loss(batch, mode="test")
        else:
            subj, logits, target = self._compute_logits(batch)

            if self.hparams.downstream_task == 'tri_class':
                output = torch.cat([logits, target.unsqueeze(1)], dim=1)
            else:
                output = torch.stack([logits.squeeze(), target.squeeze()], dim=1)

            return (subj, output.detach().cpu())

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        if not self.hparams.pretraining:
            subj_valid = []
            out_valid_list = []
            for subj, out in outputs:
                subj_valid += subj
                out_valid_list.append(out)
            subj_valid = np.array(subj_valid)
            total_out_valid = torch.cat(out_valid_list, dim=0)
            self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")

    # If you use loggers other than Neptune you may need to modify this
    def _save_predictions(self,total_subjs,total_out, mode):
        self.subject_accuracy = {}
        for subj, output in zip(total_subjs,total_out):
            if self.hparams.downstream_task == 'sex':
                score = torch.sigmoid(output[0]).item()
            else:
                score = output[0].item()

            if subj not in self.subject_accuracy:
                self.subject_accuracy[subj] = {'score': [score], 'mode':mode, 'truth':output[1], 'count':1}
            else:
                self.subject_accuracy[subj]['score'].append(score)
                self.subject_accuracy[subj]['count']+=1

        if self.hparams.strategy == None :
            pass
        elif 'ddp' in self.hparams.strategy and len(self.subject_accuracy) > 0:
            world_size = torch.distributed.get_world_size()
            total_subj_accuracy = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(total_subj_accuracy,self.subject_accuracy) # gather and broadcast to whole ranks
            accuracy_dict = {}
            for dct in total_subj_accuracy:
                for subj, metric_dict in dct.items():
                    if subj not in accuracy_dict:
                        accuracy_dict[subj] = metric_dict
                    else:
                        accuracy_dict[subj]['score']+=metric_dict['score']
                        accuracy_dict[subj]['count']+=metric_dict['count']
            self.subject_accuracy = accuracy_dict
        if self.trainer.is_global_zero:
            for subj_name,subj_dict in self.subject_accuracy.items():
                subj_pred = np.mean(subj_dict['score'])
                subj_error = np.std(subj_dict['score'])
                subj_truth = subj_dict['truth'].item()
                subj_count = subj_dict['count']
                subj_mode = subj_dict['mode'] # train, val, test

                # only save samples at rank 0 (total iterations/world_size numbers are saved)
                os.makedirs(os.path.join('predictions',self.hparams.id), exist_ok=True)
                with open(os.path.join('predictions',self.hparams.id,'iter_{}.txt'.format(self.current_epoch)),'a+') as f:
                    f.write('subject:{} ({})\ncount: {} outputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,subj_mode,subj_count,subj_pred,subj_error,subj_truth))

            with open(os.path.join('predictions',self.hparams.id,'iter_{}.pkl'.format(self.current_epoch)),'wb') as fw:
                pickle.dump(self.subject_accuracy, fw)


    def test_step(self, batch, batch_idx):
        subj, logits, target = self._compute_logits(batch)
        target = target.unsqueeze(1)
        output = torch.cat([logits, target], dim=1)
        return (subj, output)

    def test_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            self.class_names = {0: "CN", 1: "PD", 2: "Prodromal"}

            subj_test = []
            out_test_list = []
            for subj, out in outputs:
                subj_test += subj
                out_test_list.append(out.detach())
            subj_test = np.array(subj_test)
            total_out_test = torch.cat(out_test_list, dim=0)

            subjects = np.unique(subj_test)
            subj_avg_logits = []
            subj_targets = []
            misclassified_subjects = []

            for subj in subjects:
                subj_logits = total_out_test[subj_test == subj, :self.num_classes]
                subj_probs = torch.softmax(subj_logits, dim=1)
                subj_avg_probs = torch.mean(subj_probs, dim=0)
                subj_avg_logits.append(subj_avg_probs)

                true_label = total_out_test[subj_test == subj, self.num_classes][0].item()
                subj_targets.append(true_label)

                pred_label = torch.argmax(subj_avg_probs).item()
                if pred_label != true_label:
                    prob_str = ", ".join([f"{p:.4f}" for p in subj_avg_probs.detach().cpu().numpy()])

                    misclassified_subjects.append({
                        "subject": subj,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "probabilities": prob_str
                    })

            subj_avg_logits = torch.stack(subj_avg_logits)
            subj_targets = torch.tensor(subj_targets, device=total_out_test.device, dtype=torch.long)

            preds = torch.argmax(subj_avg_logits, dim=1)
            acc = (preds == subj_targets).float().mean()
            self.log(f"test_acc", acc, sync_dist=True)

            confusion = confusion_matrix(subj_targets.cpu().numpy(), preds.cpu().numpy())
            report = classification_report(subj_targets.cpu().numpy(), preds.cpu().numpy())
            print("Confusion Matrix:")
            print(confusion)
            print("Classification Report:")
            print(report)

            if misclassified_subjects:
                print("\nMisclassified Subjects:")
                for mis in misclassified_subjects:
                    print(f"Subject: {mis['subject']}")
                    print(f"  True Label: {self.class_names[mis['true_label']]} ({mis['true_label']})")
                    print(f"  Predicted Label: {self.class_names[mis['pred_label']]} ({mis['pred_label']})")
                    print(f"  Probabilities: [{mis['probabilities']}]")

                    print(f"\nTotal Misclassified Subjects: {len(misclassified_subjects)}/{len(subjects)}")

                with open("misclassified_subjects.txt", "w") as f:
                    f.write("Misclassified Subjects Report\n")
                    f.write(f"Total Misclassified: {len(misclassified_subjects)}/{len(subjects)}\n")
                    f.write("==========================================\n\n")

                    for mis in misclassified_subjects:
                        f.write(f"Subject: {mis['subject']}\n")
                        f.write(f"  True Label: {self.class_names[mis['true_label']]} ({mis['true_label']})\n")
                        f.write(f"  Predicted Label: {self.class_names[mis['pred_label']]} ({mis['pred_label']})\n")
                        f.write(f"  Probabilities: [{mis['probabilities']}]\n\n")

                        f.write("\nConfusion Matrix:\n")
                        f.write(str(confusion) + "\n\n")
                        f.write("Classification Report:\n")
                        f.write(report)

                        print("Misclassified subjects saved to 'misclassified_subjects.txt'")
            else:
                print("\nAll subjects classified correctly!")

    def on_train_epoch_start(self) -> None:

        if self.hparams.scalability_check:
            self.epoch_start_time = time.time()
            self.batch_count = 0
        return super().on_train_epoch_start()

    def on_train_batch_end(self, out, batch, batch_idx):
        if self.hparams.scalability_check:
            self.batch_count += 1
        return super().on_train_batch_end(out, batch, batch_idx)

    def on_train_epoch_end(self):
        if self.train_step_losses:
            epoch_avg_loss = torch.stack(self.train_step_losses).mean()
            self.train_epoch_losses.append(epoch_avg_loss.item())
            self.train_step_losses = []


    # def on_before_optimizer_step(self, optimizer, optimizer_idx: int) -> None:

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "SGD":
            optim = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum
            )
        else:
            print("Error: Input a correct optimizer name (default: AdamW)")

        if self.hparams.use_scheduler:
            print()
            print("training steps: " + str(self.trainer.estimated_stepping_batches))
            print("using scheduler")
            print()
            total_iterations = self.trainer.estimated_stepping_batches # ((number of samples/batch size)/number of gpus) * num_epochs
            gamma = self.hparams.gamma
            base_lr = self.hparams.learning_rate
            warmup = int(total_iterations * 0.05) # adjust the length of warmup here.
            T_0 = int(self.hparams.cycle * total_iterations)
            T_mult = 1

            sche = CosineAnnealingWarmUpRestarts(optim, first_cycle_steps=T_0, cycle_mult=T_mult, max_lr=base_lr,min_lr=1e-9, warmup_steps=warmup, gamma=gamma)
            print('total iterations:',self.trainer.estimated_stepping_batches * self.hparams.max_epochs)

            scheduler = {
                "scheduler": sche,
                "name": "lr_history",
                "interval": "step",
            }

            return [optim], [scheduler]
        else:
            return optim

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Default classifier")
        # training related
        group.add_argument("--grad_clip", action='store_true', help="whether to use gradient clipping")
        group.add_argument("--optimizer", type=str, default="AdamW", help="which optimizer to use [AdamW, SGD]")
        group.add_argument("--use_scheduler", action='store_true', help="whether to use scheduler")
        group.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for optimizer")
        group.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for optimizer")
        group.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        group.add_argument("--gamma", type=float, default=1.0, help="decay for exponential LR scheduler")
        group.add_argument("--cycle", type=float, default=0.3, help="cycle size for CosineAnnealingWarmUpRestarts")
        group.add_argument("--milestones", nargs="+", default=[100, 150], type=int, help="lr scheduler")
        group.add_argument("--adjust_thresh", action='store_true', help="whether to adjust threshold for valid/test")

        # pretraining-related
        group.add_argument("--use_contrastive", action='store_true', help="whether to use contrastive learning (specify --contrastive_type argument as well)")
        group.add_argument("--contrastive_type", default=0, type=int, help="combination of contrastive losses to use [1: Use the Instance contrastive loss function, 2: Use the local-local temporal contrastive loss function, 3: Use the sum of both loss functions]")
        group.add_argument("--pretraining", action='store_true', help="whether to use pretraining")
        group.add_argument("--augment_during_training", action='store_true', help="whether to augment input images during training")
        group.add_argument("--augment_only_affine", action='store_true', help="whether to only apply affine augmentation")
        group.add_argument("--augment_only_intensity", action='store_true', help="whether to only apply intensity augmentation")
        group.add_argument("--temperature", default=0.1, type=float, help="temperature for NTXentLoss")

        # model related
        group.add_argument("--model", type=str, default="swin4d_ver7", help="which model to be used")
        group.add_argument("--in_chans", type=int, default=1, help="Channel size of input image")
        group.add_argument("--embed_dim", type=int, default=24, help="embedding size (recommend to use 24, 36, 48)")
        group.add_argument("--window_size", nargs="+", default=[4, 4, 4, 4], type=int, help="window size from the second layers")
        group.add_argument("--first_window_size", nargs="+", default=[2, 2, 2, 2], type=int, help="first window size")
        group.add_argument("--patch_size", nargs="+", default=[5, 5, 5, 1], type=int, help="patch size")
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int, help="depth of layers in each stage")
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int, help="The number of heads for each attention layer")
        group.add_argument("--c_multiplier", type=int, default=2, help="channel multiplier for Swin Transformer architecture")
        group.add_argument("--last_layer_full_MSA", type=str2bool, default=False, help="whether to use full-scale multi-head self-attention at the last layers")
        group.add_argument("--clf_head_version", type=str, default="v1", help="clf head version, v2 has a hidden layer")
        group.add_argument("--attn_drop_rate", type=float, default=0, help="dropout rate of attention layers")

        # others
        group.add_argument("--scalability_check", action='store_true', help="whether to check scalability")
        group.add_argument("--process_code", default=None, help="Slurm code/PBS code. Use this argument if you want to save process codes to your log")
        group.add_argument("--num_classes", type=int, default=3, help="Number of classes for classification tasks")
        return parser


class LitFA3DClassifier(pl.LightningModule):
    def __init__(self, data_module, img_size, in_chans, num_classes, learning_rate=1e-4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.save_hyperparameters(kwargs)
        target_values = data_module.train_dataset.target_values
        if self.hparams.label_scaling_method == 'standardization':
            scaler = StandardScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
        elif self.hparams.label_scaling_method == 'minmax':
            scaler = MinMaxScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
        self.scaler = scaler
        self.model = SwinTransformer3D(
            img_size=img_size,
            in_chans=in_chans,
            num_classes=num_classes,
            **{k: v for k, v in kwargs.items() if k in SwinTransformer3D.__init__.__code__.co_varnames}
        )

        if not self.hparams.pretraining:
            if self.hparams.downstream_task == 'tri_class':
                self.output_head = load_model("clf_mlp", self.hparams)
        elif self.hparams.use_contrastive:
            self.output_head = load_model("emb_mlp", self.hparams)
        else:
            raise NotImplementedError("output head should be defined")

        self.metric = Metrics()
        self.learning_rate = learning_rate
        self.data_module = data_module

        if self.hparams.adjust_thresh:
            self.threshold = 0
        self.train_epoch_losses = []
        self.train_epoch_acc = []
    def forward(self, x):
        return self.output_head(self.model(x))

    def augment(self, img):
        """3D图像增强 - 仅空间维度"""
        device = img.device
        B, C, H, W, D = img.shape

        rand_affine = monai_t.RandAffine(
            prob=1.0,
            rotate_range=(0.175, 0.175, 0.175),
            scale_range=(0.1, 0.1, 0.1),
            mode="bilinear",
            padding_mode="border",
            device=device
        )
        rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
        rand_smooth = monai_t.RandGaussianSmooth(
            sigma_x=(0.0, 0.5),
            sigma_y=(0.0, 0.5),
            sigma_z=(0.0, 0.5),
            prob=0.1
        )

        if self.hparams.augment_only_intensity:
            comp = monai_t.Compose([rand_noise, rand_smooth])
        else:
            comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth])

        for b in range(B):
            aug_seed = torch.randint(0, 10000000, (1,)).item()
            comp.set_random_state(seed=aug_seed)
            img[b] = comp(img[b])

        return img

    def _compute_logits(self, batch, augment_during_training=None):
        """计算logits"""
        fa, subj, target_value, tr = batch.values()

        if augment_during_training:
            fa = self.augment(fa)

        feature = self.model(fa)
        if self.hparams.downstream_task == 'tri_class':
            logits = self.output_head(feature)
            target = target_value.float().squeeze()

        return subj, logits, target

    def _calculate_loss(self, batch, mode):
        if self.hparams.pretraining:
            loss = 0
            result_dict = {f"{mode}_loss": loss}
        else:
            subj, logits, target = self._compute_logits(
                batch,
                augment_during_training=self.hparams.augment_during_training
            )

            if self.hparams.downstream_task_type == 'classification':
                if self.hparams.downstream_task == 'tri_class':
                    loss = F.cross_entropy(logits, target.long().squeeze())
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == target.long().squeeze()).float().mean()

                    result_dict = {
                        f"{mode}_loss": loss,
                        f"{mode}_acc": acc,
                    }

        self.log_dict(
            result_dict,
            prog_bar=True,
            sync_dist=False,
            add_dataloader_idx=False,
            on_step=True,
            on_epoch=True,
            batch_size=self.hparams.batch_size
        )
        return loss

    def _evaluate_metrics(self, subj_array, total_out, mode):
        subjects = np.unique(subj_array)
        subj_avg_logits = []
        subj_targets = []

        for subj in subjects:
            if self.hparams.downstream_task == 'tri_class':
                subj_logits = total_out[subj_array == subj, :self.num_classes]
                subj_probs = torch.softmax(subj_logits, dim=1)
                subj_avg_probs = torch.mean(subj_probs, dim=0)
                subj_avg_logits.append(subj_avg_probs)
                subj_targets.append(total_out[subj_array == subj, self.num_classes][0].item())
            else:
                subj_logits = total_out[subj_array == subj, 0]
                subj_avg_logits.append(torch.mean(subj_logits).item())
                subj_targets.append(total_out[subj_array == subj, 1][0].item())

        if self.hparams.downstream_task == 'tri_class':
            subj_avg_logits = torch.stack(subj_avg_logits)
            subj_targets = torch.tensor(subj_targets, device=total_out.device, dtype=torch.long)
            preds = torch.argmax(subj_avg_logits, dim=1)
            acc = (preds == subj_targets).float().mean()
            bal_acc = balanced_accuracy_score(subj_targets.cpu().numpy(), preds.cpu().numpy())

            auroc = torchmetrics.classification.MulticlassAUROC(num_classes=self.num_classes)
            auroc_score = auroc(subj_avg_logits, subj_targets)

            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_balacc", bal_acc, sync_dist=True)
            self.log(f"{mode}_AUROC", auroc_score, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if self.hparams.pretraining:
            self._calculate_loss(batch, mode="valid")
        else:
            subj, logits, target = self._compute_logits(batch)

            if self.hparams.downstream_task == 'tri_class':
                output = torch.cat([logits, target.unsqueeze(1)], dim=1)
            else:
                output = torch.stack([logits.squeeze(), target.squeeze()], dim=1)

            return (subj, output.detach().cpu())

    def validation_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            subj_valid = []
            out_valid_list = []
            for subj, out in outputs:
                subj_valid += subj
                out_valid_list.append(out)

            subj_valid = np.array(subj_valid)
            total_out_valid = torch.cat(out_valid_list, dim=0)
            self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")

    def test_step(self, batch, batch_idx):
        subj, logits, target = self._compute_logits(batch)
        target = target.unsqueeze(1)
        output = torch.cat([logits, target], dim=1)
        return (subj, output)

    def test_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            self.class_names = {0: "CN", 1: "PD", 2: "Prodromal"}

            subj_test = []
            out_test_list = []
            for subj, out in outputs:
                subj_test += subj
                out_test_list.append(out.detach())

            subj_test = np.array(subj_test)
            total_out_test = torch.cat(out_test_list, dim=0)
            subjects = np.unique(subj_test)
            subj_avg_logits = []
            subj_targets = []
            misclassified_subjects = []

            for subj in subjects:
                subj_logits = total_out_test[subj_test == subj, :self.num_classes]
                subj_probs = torch.softmax(subj_logits, dim=1)
                subj_avg_probs = torch.mean(subj_probs, dim=0)
                subj_avg_logits.append(subj_avg_probs)

                true_label = total_out_test[subj_test == subj, self.num_classes][0].item()
                subj_targets.append(true_label)

                pred_label = torch.argmax(subj_avg_probs).item()
                if pred_label != true_label:
                    prob_str = ", ".join([f"{p:.4f}" for p in subj_avg_probs.detach().cpu().numpy()])
                    misclassified_subjects.append({
                        "subject": subj,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "probabilities": prob_str
                    })

            subj_avg_logits = torch.stack(subj_avg_logits)
            subj_targets = torch.tensor(subj_targets, device=total_out_test.device, dtype=torch.long)

            preds = torch.argmax(subj_avg_logits, dim=1)
            acc = (preds == subj_targets).float().mean()
            self.log(f"test_acc", acc, sync_dist=True)

            confusion = confusion_matrix(subj_targets.cpu().numpy(), preds.cpu().numpy())
            report = classification_report(subj_targets.cpu().numpy(), preds.cpu().numpy())
            print("Confusion Matrix:")
            print(confusion)
            print("Classification Report:")
            print(report)

            if misclassified_subjects:
                print("\nMisclassified Subjects:")
                for mis in misclassified_subjects:
                    print(f"Subject: {mis['subject']}")
                    print(f"  True Label: {self.class_names[mis['true_label']]} ({mis['true_label']})")
                    print(f"  Predicted Label: {self.class_names[mis['pred_label']]} ({mis['pred_label']})")
                    print(f"  Probabilities: [{mis['probabilities']}]")

                    print(f"\nTotal Misclassified Subjects: {len(misclassified_subjects)}/{len(subjects)}")

                with open("misclassified_subjects.txt", "w") as f:
                    f.write("Misclassified Subjects Report\n")
                    f.write(f"Total Misclassified: {len(misclassified_subjects)}/{len(subjects)}\n")
                    f.write("==========================================\n\n")

                    for mis in misclassified_subjects:
                        f.write(f"Subject: {mis['subject']}\n")
                        f.write(f"  True Label: {self.class_names[mis['true_label']]} ({mis['true_label']})\n")
                        f.write(f"  Predicted Label: {self.class_names[mis['pred_label']]} ({mis['pred_label']})\n")
                        f.write(f"  Probabilities: [{mis['probabilities']}]\n\n")

                        f.write("\nConfusion Matrix:\n")
                        f.write(str(confusion) + "\n\n")
                        f.write("Classification Report:\n")
                        f.write(report)

                        print("Misclassified subjects saved to 'misclassified_subjects.txt'")
            else:
                print("\nAll subjects classified correctly!")

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optim = torch.optim.AdamW(
                self.parameters(),
                lr=1e-3,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "SGD":
            optim = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=self.hparams.momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

        if self.hparams.use_scheduler:
            total_iterations = self.trainer.estimated_stepping_batches
            gamma = self.hparams.gamma
            base_lr = self.hparams.learning_rate
            warmup = int(total_iterations * 0.05)
            T_0 = int(self.hparams.cycle * total_iterations)
            T_mult = 1

            sche = CosineAnnealingWarmUpRestarts(
                optim,
                first_cycle_steps=T_0,
                cycle_mult=T_mult,
                max_lr=base_lr,
                min_lr=1e-9,
                warmup_steps=warmup,
                gamma=gamma
            )

            scheduler = {
                "scheduler": sche,
                "name": "lr_history",
                "interval": "step",
            }

            return [optim], [scheduler]
        else:
            return optim

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch')
        if train_loss is not None:
            self.train_epoch_losses.append(train_loss.item())

        if self.hparams.downstream_task_type == 'classification':
            train_acc = self.trainer.callback_metrics.get('train_acc_epoch')
            if train_acc is not None:
                self.train_epoch_acc.append(train_acc.item())

    def on_fit_end(self):
        if not self.trainer.is_global_zero:
            return

        os.makedirs("training_plots_fa", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._plot_train_loss_curve(timestamp)

        if self.hparams.downstream_task_type == 'classification' and self.train_epoch_acc:
            self._plot_train_acc_curve(timestamp)

    def _plot_train_loss_curve(self, timestamp):
        plt.figure(figsize=(6, 12))

        epochs = range(1, len(self.train_epoch_losses) + 1)

        plt.plot(epochs, self.train_epoch_losses, 'b-', label='Loss', linewidth=2)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if len(epochs) > 0:
            if len(epochs) <= 20:
                plt.xticks(epochs)
            else:
                step = max(1, len(epochs) // 10)
                plt.xticks(range(0, len(epochs) + 1, step))

        plt.savefig(f"training_plots_fa/train_loss_curve_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_train_acc_curve(self, timestamp):
        plt.figure(figsize=(6, 12))

        epochs = range(1, len(self.train_epoch_acc) + 1)

        plt.plot(epochs, self.train_epoch_acc, 'g-', label='Accuracy', linewidth=2)

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.ylim(0, 1.05)

        if len(epochs) > 0:
            if len(epochs) <= 20:
                plt.xticks(epochs)
            else:
                step = max(1, len(epochs) // 10)
                plt.xticks(range(0, len(epochs) + 1, step))

        plt.savefig(f"training_plots_fa/train_accuracy_curve_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Default classifier")
        # training related
        group.add_argument("--grad_clip", action='store_true', help="whether to use gradient clipping")
        group.add_argument("--optimizer", type=str, default="AdamW", help="which optimizer to use [AdamW, SGD]")
        group.add_argument("--use_scheduler", action='store_true', help="whether to use scheduler")
        group.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for optimizer")
        group.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for optimizer")
        group.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        group.add_argument("--gamma", type=float, default=1.0, help="decay for exponential LR scheduler")
        group.add_argument("--cycle", type=float, default=0.3, help="cycle size for CosineAnnealingWarmUpRestarts")
        group.add_argument("--milestones", nargs="+", default=[100, 150], type=int, help="lr scheduler")
        group.add_argument("--adjust_thresh", action='store_true', help="whether to adjust threshold for valid/test")

        # pretraining-related
        group.add_argument("--use_contrastive", action='store_true',
                           help="whether to use contrastive learning (specify --contrastive_type argument as well)")
        group.add_argument("--contrastive_type", default=0, type=int,
                           help="combination of contrastive losses to use [1: Instance contrastive, 2: Local-local, 3: Both]")
        group.add_argument("--pretraining", action='store_true', help="whether to use pretraining")
        group.add_argument("--augment_during_training", action='store_true',
                           help="whether to augment input images during training")
        group.add_argument("--augment_only_affine", action='store_true',
                           help="whether to only apply affine augmentation")
        group.add_argument("--augment_only_intensity", action='store_true', default=False,
                           help="whether to only apply intensity augmentation")
        group.add_argument("--temperature", default=0.1, type=float, help="temperature for NTXentLoss")

        # model related
        group.add_argument("--model", type=str, default="swin3d_transformer", help="which model to be used")
        group.add_argument("--in_chans", type=int, default=1, help="Channel size of input image")
        group.add_argument("--embed_dim", type=int, default=24, help="embedding size (24, 36, 48)")
        group.add_argument("--window_size", nargs="+", default=[4, 4, 4], type=int,
                           help="window size for Swin3D")
        group.add_argument("--first_window_size", nargs="+", default=[2, 2, 2], type=int,
                           help="first window size for Swin3D")
        group.add_argument("--patch_size", nargs="+", default=[2, 2, 2], type=int, help="patch size")#6 6 6
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int,
                           help="depth of layers in each stage")
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int,
                           help="The number of heads for each attention layer")
        group.add_argument("--c_multiplier", type=int, default=2,
                           help="channel multiplier for Swin Transformer architecture")
        group.add_argument("--last_layer_full_MSA", type=str2bool, default=False,
                           help="whether to use full-scale MSA at the last layers")
        group.add_argument("--clf_head_version", type=str, default="v1",
                           help="clf head version (v1 or v2)")
        group.add_argument("--attn_drop_rate", type=float, default=0,
                           help="dropout rate of attention layers")

        # others
        group.add_argument("--scalability_check", action='store_true', help="whether to check scalability")
        group.add_argument("--process_code", default=None,
                           help="Slurm/PBS code for logging")

        group.add_argument("--num_classes", type=int, default=3,
                           help="Number of classes for classification tasks")
        return parser


class LitMultimodalClassifier(pl.LightningModule):
    def __init__(self, data_module, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.save_hyperparameters(kwargs)

        if self.hparams.label_scaling_method == 'standardization':
            scaler = StandardScaler()
            #print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
        elif self.hparams.label_scaling_method == 'minmax':
            scaler = MinMaxScaler()
            #print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
        self.scaler = scaler

        self.fmri_model = load_model("swin4d_ver7", self.hparams)  # 4D fMRI模型
        self.fa_model = load_model("swin3d_transformer", self.hparams)  # 3D FA模型

        if not self.hparams.pretraining:
            if self.hparams.downstream_task == 'tri_class':
                fusion_dim = self.fmri_model.output_dim + self.fa_model.output_dim
                self.output_head = load_model("clf_mlp_mulit", self.hparams)
        elif self.hparams.use_contrastive:
            fusion_dim = self.fmri_model.output_dim + self.fa_model.output_dim
            self.output_head = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        else:
            raise NotImplementedError("output head should be defined")

        self.metric = Metrics()
        self.fusion_module = AxialAttentionTransformer(dim=48, depth=2, heads=8, reversible=True)
        self.train_epoch_losses = []
        self.train_step_losses = []


        if self.hparams.adjust_thresh:
            self.threshold = 0

    def forward(self, fmri, fa):

        fmri_feat = self.fmri_model(fmri)  # [B, D1]
        fa_feat = self.fa_model(fa)  # [B, D2]
        fusion_feat = optimized_fusion(fmri_feat, fa_feat)

        return self.output_head(fusion_feat)

    def _compute_logits(self, batch, augment_during_training=None):
        """计算logits"""
        fmri = batch["fmri_sequence"]
        fa = batch["fa_sequence"]
        subj = batch["subject_name"]
        target_value = batch["target"]
        tr = batch["TR"]

        if self.hparams.downstream_task == 'tri_class':
            logits = self.forward(fmri, fa)
            target = target_value.float().squeeze()

        return subj, logits, target
    def on_fit_end(self):

        if not self.trainer.is_global_zero:
            return

        plt.figure(figsize=(10, 6))

        epochs = range(len(self.train_epoch_losses))

        plt.plot(epochs, self.train_epoch_losses, 'b-', label='Training Loss', linewidth=2)

        plt.title('Training Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        if len(self.train_epoch_losses) > 0:
            if len(self.train_epoch_losses) <= 20:
                plt.xticks(range(len(self.train_epoch_losses)))
            else:
                step = max(1, len(self.train_epoch_losses) // 10)
                plt.xticks(range(0, len(self.train_epoch_losses), step))

        os.makedirs("training_loss_plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"training_loss_plots_Combine/training_loss_per_epoch_{timestamp}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training loss curve saved to training_loss_plots/training_loss_per_epoch_{timestamp}.png")
    def _calculate_loss(self, batch, mode):

        if self.hparams.pretraining and self.hparams.use_contrastive:
            fmri, fa, subj, target_value, tr = batch.values()
            y, diff_y = fmri

            batch_size = y.shape[0]
            criterion = NTXentLoss(device='cuda', batch_size=batch_size,
                                   temperature=self.hparams.temperature,
                                   use_cosine_similarity=True).cuda()

            out_y = self.forward(y, fa)
            out_diff_y = self.forward(diff_y, fa)

            loss = criterion(out_y, out_diff_y)

            result_dict = {
                f"{mode}_loss": loss,
            }
        else:
            subj, logits, target = self._compute_logits(
                batch,
                augment_during_training=self.hparams.augment_during_training
            )

            if self.hparams.downstream_task_type == 'classification':
                if self.hparams.downstream_task == 'tri_class':
                    loss = F.cross_entropy(logits, target.long().squeeze())#4 48 4 3， 1 2 2 0（label batch：4）
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == target.long().squeeze()).float().mean()

                    result_dict = {
                        f"{mode}_loss": loss,
                        f"{mode}_acc": acc,
                    }

        self.log_dict(
            result_dict,
            prog_bar=True,
            sync_dist=False,
            add_dataloader_idx=False,
            on_step=True,
            on_epoch=True,
            batch_size=self.hparams.batch_size
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        self.train_step_losses.append(loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if self.hparams.pretraining:
            if dataloader_idx == 0:
                self._calculate_loss(batch, mode="valid")
            else:
                self._calculate_loss(batch, mode="test")
        else:
            subj, logits, target = self._compute_logits(batch)

            if self.hparams.downstream_task == 'tri_class':
                output = torch.cat([logits, target.unsqueeze(1)], dim=1)
            else:
                output = torch.stack([logits.squeeze(), target.squeeze()], dim=1)

            return (subj, output.detach().cpu())

    def validation_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            subj_valid = []
            out_valid_list = []
            for subj, out in outputs:
                subj_valid += subj
                out_valid_list.append(out)

            subj_valid = np.array(subj_valid)
            total_out_valid = torch.cat(out_valid_list, dim=0)
            self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")

    def _evaluate_metrics(self, subj_array, total_out, mode):
        subjects = np.unique(subj_array)
        subj_avg_logits = []
        subj_targets = []

        for subj in subjects:
            if self.hparams.downstream_task == 'tri_class':
                subj_logits = total_out[subj_array == subj, :self.num_classes]
                subj_probs = torch.softmax(subj_logits, dim=1)
                subj_avg_probs = torch.mean(subj_probs, dim=0)
                subj_avg_logits.append(subj_avg_probs)
                subj_targets.append(total_out[subj_array == subj, self.num_classes][0].item())
            else:
                subj_logits = total_out[subj_array == subj, 0]
                subj_avg_logits.append(torch.mean(subj_logits).item())
                subj_targets.append(total_out[subj_array == subj, 1][0].item())

        if self.hparams.downstream_task == 'tri_class':
            subj_avg_logits = torch.stack(subj_avg_logits)
            subj_targets = torch.tensor(subj_targets, device=total_out.device, dtype=torch.long)

            preds = torch.argmax(subj_avg_logits, dim=1)
            acc = (preds == subj_targets).float().mean()
            bal_acc = balanced_accuracy_score(subj_targets.cpu().numpy(), preds.cpu().numpy())

            auroc = torchmetrics.classification.MulticlassAUROC(num_classes=self.num_classes)
            auroc_score = auroc(subj_avg_logits, subj_targets)

            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_balacc", bal_acc, sync_dist=True)
            self.log(f"{mode}_AUROC", auroc_score, sync_dist=True)
        elif self.hparams.downstream_task == 'sex' or self.hparams.scalability_check:
            subj_avg_logits = torch.tensor(subj_avg_logits, device=total_out.device)
            subj_targets = torch.tensor(subj_targets, device=total_out.device)

            acc_func = BinaryAccuracy().to(total_out.device)
            auroc_func = BinaryAUROC().to(total_out.device)

            acc = acc_func((subj_avg_logits >= 0).int(), subj_targets)
            bal_acc_sk = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits >= 0).int().cpu())
            auroc = auroc_func(torch.sigmoid(subj_avg_logits), subj_targets)

            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_balacc", bal_acc_sk, sync_dist=True)
            self.log(f"{mode}_AUROC", auroc, sync_dist=True)
        elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task_type == 'regression':
            subj_avg_logits = torch.tensor(subj_avg_logits, device=total_out.device)
            subj_targets = torch.tensor(subj_targets, device=total_out.device)

            mse = F.mse_loss(subj_avg_logits, subj_targets)
            mae = F.l1_loss(subj_avg_logits, subj_targets)

            if self.hparams.label_scaling_method == 'standardization':
                adjusted_mse = F.mse_loss(
                    subj_avg_logits * self.scaler.scale_[0] + self.scaler.mean_[0],
                    subj_targets * self.scaler.scale_[0] + self.scaler.mean_[0]
                )
                adjusted_mae = F.l1_loss(
                    subj_avg_logits * self.scaler.scale_[0] + self.scaler.mean_[0],
                    subj_targets * self.scaler.scale_[0] + self.scaler.mean_[0]
                )
            elif self.hparams.label_scaling_method == 'minmax':
                adjusted_mse = F.mse_loss(
                    subj_avg_logits * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0],
                    subj_targets * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0]
                )
                adjusted_mae = F.l1_loss(
                    subj_avg_logits * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0],
                    subj_targets * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0]
                )

            pearson = PearsonCorrCoef().to(total_out.device)
            pearson_coef = pearson(subj_avg_logits, subj_targets)

            self.log(f"{mode}_corrcoef", pearson_coef, sync_dist=True)
            self.log(f"{mode}_mse", mse, sync_dist=True)
            self.log(f"{mode}_mae", mae, sync_dist=True)
            self.log(f"{mode}_adjusted_mse", adjusted_mse, sync_dist=True)
            self.log(f"{mode}_adjusted_mae", adjusted_mae, sync_dist=True)

    def test_step(self, batch, batch_idx):
        subj, logits, target = self._compute_logits(batch)
        target = target.unsqueeze(1)
        output = torch.cat([logits, target], dim=1)
        return (subj, output)

    def test_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            self.class_names = {0: "CN", 1: "PD", 2: "Prodromal"}

            subj_test = []
            out_test_list = []
            for subj, out in outputs:
                subj_test += subj
                out_test_list.append(out.detach())

            subj_test = np.array(subj_test)
            total_out_test = torch.cat(out_test_list, dim=0)

            subjects = np.unique(subj_test)
            subj_avg_logits = []
            subj_targets = []
            misclassified_subjects = []

            for subj in subjects:
                subj_logits = total_out_test[subj_test == subj, :self.num_classes]
                subj_probs = torch.softmax(subj_logits, dim=1)
                subj_avg_probs = torch.mean(subj_probs, dim=0)
                subj_avg_logits.append(subj_avg_probs)

                true_label = total_out_test[subj_test == subj, self.num_classes][0].item()
                subj_targets.append(true_label)

                pred_label = torch.argmax(subj_avg_probs).item()
                if pred_label != true_label:
                    prob_str = ", ".join([f"{p:.4f}" for p in subj_avg_probs.detach().cpu().numpy()])
                    misclassified_subjects.append({
                        "subject": subj,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "probabilities": prob_str
                    })

            subj_avg_logits = torch.stack(subj_avg_logits)
            subj_targets = torch.tensor(subj_targets, device=total_out_test.device, dtype=torch.long)

            preds = torch.argmax(subj_avg_logits, dim=1)
            acc = (preds == subj_targets).float().mean()
            self.log(f"test_acc", acc, sync_dist=True)

            confusion = confusion_matrix(subj_targets.cpu().numpy(), preds.cpu().numpy())
            report = classification_report(subj_targets.cpu().numpy(), preds.cpu().numpy())
            print("Confusion Matrix:")
            print(confusion)
            print("Classification Report:")
            print(report)

            if misclassified_subjects:
                print("\nMisclassified Subjects:")
                for mis in misclassified_subjects:
                    print(f"Subject: {mis['subject']}")
                    print(f"  True Label: {self.class_names[mis['true_label']]} ({mis['true_label']})")
                    print(f"  Predicted Label: {self.class_names[mis['pred_label']]} ({mis['pred_label']})")
                    print(f"  Probabilities: [{mis['probabilities']}]")

                    print(f"\nTotal Misclassified Subjects: {len(misclassified_subjects)}/{len(subjects)}")

                with open("misclassified_subjects.txt", "w") as f:
                    f.write("Misclassified Subjects Report\n")
                    f.write(f"Total Misclassified: {len(misclassified_subjects)}/{len(subjects)}\n")
                    f.write("==========================================\n\n")

                    for mis in misclassified_subjects:
                        f.write(f"Subject: {mis['subject']}\n")
                        f.write(f"  True Label: {self.class_names[mis['true_label']]} ({mis['true_label']})\n")
                        f.write(f"  Predicted Label: {self.class_names[mis['pred_label']]} ({mis['pred_label']})\n")
                        f.write(f"  Probabilities: [{mis['probabilities']}]\n\n")

                        f.write("\nConfusion Matrix:\n")
                        f.write(str(confusion) + "\n\n")
                        f.write("Classification Report:\n")
                        f.write(report)

                        print("Misclassified subjects saved to 'misclassified_subjects.txt'")
            else:
                print("\nAll subjects classified correctly!")

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optim = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "SGD":
            optim = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=self.hparams.momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

        if self.hparams.use_scheduler:
            total_iterations = self.trainer.estimated_stepping_batches
            gamma = self.hparams.gamma
            base_lr = self.hparams.learning_rate
            warmup = int(total_iterations * 0.05)
            T_0 = int(self.hparams.cycle * total_iterations)
            T_mult = 1

            sche = CosineAnnealingWarmUpRestarts(
                optim,
                first_cycle_steps=T_0,
                cycle_mult=T_mult,
                max_lr=base_lr,
                min_lr=1e-9,
                warmup_steps=warmup,
                gamma=gamma
            )

            scheduler = {
                "scheduler": sche,
                "name": "lr_history",
                "interval": "step",
            }

            return [optim], [scheduler]
        else:
            return optim
    def on_train_epoch_end(self):
        if self.train_step_losses:
            epoch_avg_loss = torch.stack(self.train_step_losses).mean()
            self.train_epoch_losses.append(epoch_avg_loss.item())
            self.train_step_losses = []
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Multimodal classifier")

        # 训练相关参数
        group.add_argument("--grad_clip", action='store_true', help="是否使用梯度裁剪")
        group.add_argument("--optimizer", type=str, default="AdamW", help="优化器类型 [AdamW, SGD]")
        group.add_argument("--use_scheduler", action='store_true', help="是否使用学习率调度器")
        group.add_argument("--weight_decay", type=float, default=0.01, help="优化器权重衰减")
        group.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
        group.add_argument("--momentum", type=float, default=0, help="SGD动量")
        group.add_argument("--gamma", type=float, default=1.0, help="指数学习率调度器的衰减率")
        group.add_argument("--cycle", type=float, default=0.3, help="CosineAnnealingWarmUpRestarts的周期大小")
        group.add_argument("--milestones", nargs="+", default=[100, 150], type=int, help="学习率里程碑")
        group.add_argument("--adjust_thresh", action='store_true', help="是否调整验证/测试的阈值")

        # 预训练相关
        group.add_argument("--use_contrastive", action='store_true', help="是否使用对比学习")
        group.add_argument("--contrastive_type", default=0, type=int,
                           help="对比损失组合 [1: 实例对比, 2: 局部-局部对比, 3: 两者结合]")
        group.add_argument("--pretraining", action='store_true', help="是否使用预训练")
        group.add_argument("--augment_during_training", action='store_true',
                           help="训练期间是否增强输入图像")
        group.add_argument("--augment_only_affine", action='store_true',
                           help="是否仅应用仿射增强")
        group.add_argument("--augment_only_intensity", action='store_true',
                           help="是否仅应用强度增强")
        group.add_argument("--temperature", default=0.1, type=float,
                           help="NTXentLoss的温度参数")

        # 模型相关
        group.add_argument("--model", type=str, default="swin4d_ver7", help="主模型类型")
        group.add_argument("--in_chans", type=int, default=1, help="输入图像通道数")
        group.add_argument("--embed_dim", type=int, default=24, help="嵌入维度 (推荐24, 36, 48)")
        group.add_argument("--window_size", nargs="+", default=[4, 4, 4, 4], type=int,
                           help="第二层起的窗口大小")
        group.add_argument("--first_window_size", nargs="+", default=[2, 2, 2, 2], type=int,
                           help="第一层窗口大小")
        group.add_argument("--patch_size", nargs="+", default=[5, 5, 5, 1], type=int,
                           help="patch大小")
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int,
                           help="各阶段的层深度")
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int,
                           help="注意力层的头数")
        group.add_argument("--c_multiplier", type=int, default=2,
                           help="Swin Transformer架构的通道乘数")
        group.add_argument("--last_layer_full_MSA", type=str2bool, default=False,
                           help="最后一层是否使用全尺度多头自注意力")
        group.add_argument("--clf_head_version", type=str, default="v1",
                           help="分类头版本，v2有隐藏层")
        group.add_argument("--attn_drop_rate", type=float, default=0,
                           help="注意力层的dropout率")

        # FA模型特有参数
        group.add_argument("--fa_embed_dim", type=int, default=24,
                           help="FA模型的嵌入维度")
        group.add_argument("--fa_window_size", nargs="+", default=[4, 4, 4], type=int,
                           help="FA模型的窗口大小")
        group.add_argument("--fa_first_window_size", nargs="+", default=[2, 2, 2], type=int,
                           help="FA模型的第一层窗口大小")
        group.add_argument("--fa_patch_size", nargs="+", default=[2, 2, 2], type=int,
                           help="FA模型的patch大小")

        # 其他
        group.add_argument("--scalability_check", action='store_true',
                           help="是否进行可扩展性检查")
        group.add_argument("--process_code", default=None,
                           help="Slurm/PBS代码，用于日志记录")

        # 分类任务参数
        group.add_argument("--num_classes", type=int, default=3,
                           help="分类任务的类别数")

        return parser