import os
import numpy as np
import yaml
import argparse
import torch
import torchmetrics
from time import time, sleep
from tqdm import tqdm
from utils.utils import load_configs_infer, test_gpu_cuda, prepare_tensorboard, prepare_optimizer_infer, save_checkpoint_infer, \
    get_logging, load_checkpoints_infer, prepare_saving_dir
from data.data_idr import prepare_dataloaders_idr_kfold
from model_idr import prepare_models
from accelerate import Accelerator
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
# from focal_loss import FocalLoss

def compute_idr_metrics(preds_list, labels_list):
    """Calculates AUC, APS, and Max F1 for valid residue positions."""
    if not preds_list: return 0, 0, 0
    
    y_scores = torch.cat(preds_list).cpu().numpy()
    y_true = torch.cat(labels_list).cpu().numpy()
    
    # Basic metrics
    auc = roc_auc_score(y_true, y_scores)
    aps = average_precision_score(y_true, y_scores)
    
    # Calculate Max F1 across multiple thresholds
    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1 = 0
    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            
    return auc, aps, best_f1

def train(accelerator, dataloader, tools, global_step):
    # Initialize metrics
    # accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=tools['num_classes'])
    # f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average='macro', task="multiclass")
    # accuracy.to(accelerator.device)
    # f1_score.to(accelerator.device)
    tools['net'].train()
    tools["optimizer"].zero_grad()
    epoch_loss = 0
    train_loss = 0
    counter = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(range(global_step, int(np.ceil(len(dataloader) / tools['accum_iter']))),
                        disable=not accelerator.is_local_main_process, leave=False)
    progress_bar.set_description("Steps")
    for i, data in enumerate(dataloader):
        with accelerator.accumulate(tools['net']):
            # sequence, labels, sample_weight = data
            ids, seqs, labels = data
            logits = tools['net'](seqs)
            labels_tensor = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1).to(accelerator.device)
            mask = (labels_tensor != -1)
            loss = tools['loss_function'](logits[mask], labels_tensor[mask].float())

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(tools['net'].parameters(), tools['grad_clip'])
            tools['optimizer'].step()
            tools['scheduler'].step()
            tools['optimizer'].zero_grad()
        if accelerator.sync_gradients:
            # if tensorboard_log:
            #     tools['train_writer'].add_scalar('step loss', train_loss, global_step)
            #     tools['train_writer'].add_scalar('learning rate', tools['optimizer'].param_groups[0]['lr'], global_step)
            # progress_bar.update(1)
            global_step += 1
            counter += 1
            epoch_loss += loss.item()

            probs = torch.sigmoid(logits[mask]).detach()
            all_preds.append(accelerator.gather(probs))
            all_labels.append(accelerator.gather(labels_tensor[mask]))
            # train_loss = 0
        # logs = {"step_loss": loss.detach().item(),
        #         "lr": tools['optimizer'].param_groups[0]['lr']}
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})
    # epoch_acc = accuracy.compute().cpu().item()
    # epoch_f1 = f1_score.compute().cpu().item()
    # epoch_spearman_cor = spearmanr(pred_list, label_list)
    # Aggregate metrics at the end of the epoch
    metrics = compute_idr_metrics(all_preds, all_labels)
    return epoch_loss / counter, metrics


def valid(accelerator, dataloader, tools):
    tools['net'].eval()
    # Initialize metrics
    # accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=tools['num_classes'])
    # macro_f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average='macro', task="multiclass")
    # f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average=None, task="multiclass")
    # accuracy.to(accelerator.device)
    # macro_f1_score.to(accelerator.device)
    # f1_score.to(accelerator.device)
    counter = 0
    all_preds = []
    all_labels = []
    valid_loss = 0
    # progress_bar = tqdm(range(len(dataloader)),
    #                     disable=not accelerator.is_local_main_process, leave=False)
    # progress_bar.set_description("Steps")
    valid_loss = 0
    for i, data in enumerate(dataloader):
        # sequence, labels, sample_weight = data
        ids, seqs, labels = data
        labels_tensor = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1).to(accelerator.device)
        mask = (labels_tensor != -1)
        with torch.inference_mode():
            logits = tools['net'](seqs)
            loss = tools['loss_function'](logits[mask], labels_tensor[mask].float())
            # pred_list.extend(outputs)
            # label_list.extend(ddg)
            probs = torch.sigmoid(logits[mask])
            all_preds.append(accelerator.gather(probs))
            all_labels.append(accelerator.gather(labels_tensor[mask]))
            valid_loss += loss.item()
            counter += 1
            # preds = torch.argmax(outputs, dim=1)
            # accuracy.update(accelerator.gather(preds).detach(), accelerator.gather(labels).detach())
            # macro_f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(labels).detach())
            # f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(labels).detach())
    metrics = compute_idr_metrics(all_preds, all_labels)
    return valid_loss / counter, metrics


def main(args, dict_config, config_file_path):
    configs = load_configs_infer(dict_config, args)
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)
    torch.cuda.empty_cache()
    test_gpu_cuda()

    N_FOLDS = 5
    fold_best_paths = []
    
    for fold in range(N_FOLDS):
        # 1) create fold-specific output directory
        result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)
        fold_result_path = os.path.join(result_path, f"fold_{fold}")
        logging = get_logging(result_path)
        os.makedirs(fold_result_path, exist_ok=True)
    
        # you likely also want separate checkpoint subdir
        fold_checkpoint_path = os.path.join(fold_result_path, "checkpoints")
        os.makedirs(fold_checkpoint_path, exist_ok=True)
    
        # 2) (re)create Accelerator per fold (cleanest)
        accelerator = Accelerator(
            mixed_precision=configs.train_settings.mixed_precision,
            gradient_accumulation_steps=configs.train_settings.grad_accumulation,
            dispatch_batches=False
        )
        if accelerator.is_main_process:
            accelerator.init_trackers(f"fold_{fold}", config=None)
    
        # 3) build fold dataloaders
        dataloaders_dict = prepare_dataloaders_idr_kfold(configs, fold_idx=fold, n_splits=N_FOLDS, seed=42)
    
        # 4) init model/opt/sched
        net = prepare_models(configs, logging)
        optimizer, scheduler = prepare_optimizer_infer(net, configs, len(dataloaders_dict["train"]), logging)
        net, start_epoch = load_checkpoints_infer(configs, optimizer, scheduler, logging, net)
        # 5) accelerator.prepare
        dataloaders_dict["train"], dataloaders_dict["valid"], dataloaders_dict["test"] = accelerator.prepare(
            dataloaders_dict["train"],
            dataloaders_dict["valid"],
            dataloaders_dict["test"]
        )
        net, optimizer, scheduler = accelerator.prepare(net, optimizer, scheduler)
    
        # 6) tools dict must use fold paths
        # tools = {... same as before ...}
        if configs.train_settings.loss == 'mse':
            criterion = F.mse_loss
        elif configs.train_settings.loss == 'BCE':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            print('wrong optimizer!')
            exit()
        tools = {
        'net': net,
        'train_device': configs.train_settings.device,
        'valid_device': configs.valid_settings.device,
        'train_batch_size': configs.train_settings.batch_size,
        'valid_batch_size': configs.valid_settings.batch_size,
        'optimizer': optimizer,
        'mixed_precision': configs.train_settings.mixed_precision,
        # 'train_writer': train_writer,
        # 'valid_writer': valid_writer,
        'accum_iter': configs.train_settings.grad_accumulation,
        'loss_function': criterion,
        'grad_clip': configs.optimizer.grad_clip_norm,
        'checkpoints_every': configs.checkpoints_every,
        'scheduler': scheduler,
        'result_path': result_path,
        'checkpoint_path': checkpoint_path,
        'logging': logging,
        'num_classes': configs.encoder.num_classes
        }
        tools["result_path"] = fold_result_path
        tools["checkpoint_path"] = fold_checkpoint_path
    
        # 7) train epochs as you do now, saving best for this fold
        best_valid_loss = np.inf
        best_valid_auc = 0
        global_step = 0
        for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
            tools['epoch'] = epoch
            start_time = time()
            train_loss, train_metrics = train(accelerator, dataloaders_dict["train"],
                                                    tools, global_step)
            end_time = time()
            if accelerator.is_main_process:
                logging.info(f'epoch {epoch} - time {np.round(end_time - start_time, 2)}s, '
                             f'train loss {np.round(train_loss, 4)}, train auc {np.round(train_metrics[0], 4)}, '
                             f'train aps {np.round(train_metrics[1], 4)}, train max f1 {np.round(train_metrics[2], 4)}')
    
            if epoch % configs.valid_settings.do_every == 0 and epoch != 0:
                start_time = time()
                valid_loss, valid_metrics = valid(accelerator,
                                              dataloaders_dict["valid"],
                                              tools)
                end_time = time()
                if accelerator.is_main_process:
                    logging.info(
                        f'evaluation - time {np.round(end_time - start_time, 2)}s, '
                        f'valid loss {np.round(valid_loss, 6)}, valid auc {np.round(valid_metrics[0], 6)}, '
                        f'valid aps {np.round(valid_metrics[1], 4)}, valid max f1 {np.round(valid_metrics[2], 4)}')
    
                if valid_loss< best_valid_loss:
                    best_valid_loss = valid_loss
                    # model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model_loss.pth')
                    model_path = os.path.join(fold_checkpoint_path, "best_model_loss.pth")
                    accelerator.wait_for_everyone()
                    save_checkpoint_infer(epoch, model_path, tools, accelerator)
                if valid_metrics[0] > best_valid_auc:
                    best_valid_auc = valid_metrics[0]
                    # model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model_auc.pth')
                    model_path = os.path.join(fold_checkpoint_path, "best_model_auc.pth")
                    save_checkpoint_infer(epoch, model_path, tools, accelerator)
                
    
            # if epoch % configs.checkpoints_every == 0:
            #     # Set the path to save the model checkpoint.
            #     model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')
            #     accelerator.wait_for_everyone()
            #     save_checkpoint(epoch, model_path, tools, accelerator)
    
        if accelerator.is_main_process:
            logging.info(f'best valid mse: {np.round(best_valid_loss, 6)}')
            logging.info(f'best valid auc: {np.round(best_valid_auc, 6)}')
    
        accelerator.end_training()
        accelerator.free_memory()
        del tools, net, dataloaders_dict, optimizer, scheduler, accelerator
        torch.cuda.empty_cache()

def predict_probs(accelerator, dataloader, net):
    net.eval()
    all_probs = []
    all_labels = []
    for ids, seqs, labels in dataloader:
        labels_tensor = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1).to(accelerator.device)
        mask = (labels_tensor != -1)
        with torch.inference_mode():
            logits = net(seqs)  # [B, L] or [B, L, 1] depending on your net
            probs = torch.sigmoid(logits)
        # flatten masked positions in a consistent order
        probs_1d = probs[mask].detach()
        labels_1d = labels_tensor[mask].detach()
        all_probs.append(accelerator.gather(probs_1d))
        all_labels.append(accelerator.gather(labels_1d))
    return torch.cat(all_probs), torch.cat(all_labels)

def test_evaluation(args, dict_config, config_file_path):
    configs = load_configs_infer(dict_config, args)
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)
    torch.cuda.empty_cache()
    test_gpu_cuda()
    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)
    logging = get_logging(result_path)
    # Build ONE accelerator for eval (or plain torch, but keep consistent)
    accelerator = Accelerator(mixed_precision=configs.train_settings.mixed_precision)
    # prepare the test loader once (same test set every fold)
    test_loaders = prepare_dataloaders_idr_kfold(configs, fold_idx=0, n_splits=5, seed=42)
    test_loader = accelerator.prepare(test_loaders["test"])
    probs_folds = []
    labels_ref = None
    for fold, ckpt_path in enumerate([0,0,0,0,0]):
        result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)
        fold_result_path = os.path.join(result_path, f"fold_{fold}")
        fold_checkpoint_path = os.path.join(fold_result_path, "checkpoints", 'best_model_loss.pth')
        net = prepare_models(configs, logging)
        ckpt = torch.load(fold_checkpoint_path, map_location="cpu")
        net.load_state_dict(ckpt["model_state_dict"])
        net = accelerator.prepare(net)
        probs, labels = predict_probs(accelerator, test_loader, net)
        probs_folds.append(probs.cpu())
        if labels_ref is None:
            labels_ref = labels.cpu()
        else:
            # sanity check: labels order must match
            assert torch.equal(labels_ref, labels.cpu())
    # average probs across folds
    probs_ens = torch.stack(probs_folds, dim=0).mean(dim=0)
    # compute metrics
    auc, aps, max_f1 = compute_idr_metrics([probs_ens], [labels_ref])
    logging.info(f"Ensemble test AUC={auc:.4f} APS={aps:.4f} MaxF1={max_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classification model using esm")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./config_fold.yaml')
    parser.add_argument("--result_path", default=None,
                        help="result_path, if setted by command line, "
                             "overwrite the one in config.yaml, by default is None")
    parser.add_argument("--resume_path", default=None,
                        help="if set, overwrite the one in config.yaml, by default is None")
    parser.add_argument("--num_end_adapter_layers", default=None, help="num_end_adapter_layers")
    parser.add_argument("--module_type", default=None, help="module_type for adapterh")
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(args, config_file, config_path)
    test_evaluation(args, config_file, config_path)
    print('done!')