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
from data.data_idr import prepare_dataloaders_idr
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
    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)
    logging = get_logging(result_path)
    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        # split_batches=True,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        dispatch_batches=False
    )
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("accelerator_tracker", config=None)
    dataloaders_dict = prepare_dataloaders_idr(configs)
    logging.info('preparing dataloaders are done')
    net = prepare_models(configs, logging)
    logging.info('preparing model is done')
    optimizer, scheduler = prepare_optimizer_infer(net, configs, len(dataloaders_dict["train"]), logging)
    logging.info('preparing optimizer is done')
    net, start_epoch = load_checkpoints_infer(configs, optimizer, scheduler, logging, net)
    dataloaders_dict["train"], dataloaders_dict["valid"], dataloaders_dict['test'] = accelerator.prepare(
        dataloaders_dict["train"],
        dataloaders_dict["valid"],
        dataloaders_dict['test']
    )
    # dataloaders_dict["test_fold"], dataloaders_dict["test_family"], dataloaders_dict[
    #     "test_super_family"] = accelerator.prepare(
    #     dataloaders_dict["test_fold"],
    #     dataloaders_dict["test_family"],
    #     dataloaders_dict["test_super_family"],
    # )
    net, optimizer, scheduler = accelerator.prepare(
        net, optimizer, scheduler
    )
    # initialize tensorboards
    train_writer, valid_writer = prepare_tensorboard(result_path)
    # prepare loss function
    # if configs.train_settings.loss == 'crossentropy':
    #     criterion = torch.nn.CrossEntropyLoss(reduction="none")
    # elif configs.train_settings.loss == 'focal':
    #     alpha_value = torch.full((configs.encoder.num_classes,), 0.8)
    #     criterion = FocalLoss(
    #         alpha=alpha_value,
    #         gamma=2.0,
    #         reduction='none')
    #     criterion = accelerator.prepare(criterion)
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
        'train_writer': train_writer,
        'valid_writer': valid_writer,
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
    logging.info(f'number of train steps per epoch: {np.ceil(len(dataloaders_dict["train"]) / tools["accum_iter"])}')
    logging.info(f'number of valid steps per epoch: {len(dataloaders_dict["valid"])}')
    logging.info(f'number of test steps per epoch: {len(dataloaders_dict["test"])}')
    # logging.info(f'number of test_family steps per epoch: {len(dataloaders_dict["test_family"])}')
    # logging.info(f'number of test_super_family steps per epoch: {len(dataloaders_dict["test_super_family"])}')
    # best_valid_acc = 0
    # best_valid_f1 = {}
    # best_valid_macro_f1 = 0
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
                model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model_loss.pth')
                accelerator.wait_for_everyone()
                save_checkpoint_infer(epoch, model_path, tools, accelerator)
            if valid_metrics[0] > best_valid_auc:
                best_valid_auc = valid_metrics[0]
                model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model_auc.pth')
                save_checkpoint_infer(epoch, model_path, tools, accelerator)
            

        # if epoch % configs.checkpoints_every == 0:
        #     # Set the path to save the model checkpoint.
        #     model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')
        #     accelerator.wait_for_everyone()
        #     save_checkpoint(epoch, model_path, tools, accelerator)

    if accelerator.is_main_process:
        logging.info(f'best valid mse: {np.round(best_valid_loss, 6)}')
        logging.info(f'best valid auc: {np.round(best_valid_auc, 6)}')
        # logging.info(f'best valid aps: {np.round(best_valid_spearman, 6)}')
        # logging.info(f'best valid max f1: {np.round(best_valid_spearman, 6)}')
        # best_valid_f1 = [np.round(v, 4) for v in best_valid_f1.tolist()]
        # logging.info(f'best valid f1: {best_valid_f1}')

    train_writer.close()
    valid_writer.close()

    # pause 20 second to make sure the best validation checkpoint is ready on the disk
    sleep(20)

    model_path = os.path.join(tools['result_path'], 'checkpoints', 'best_model_loss.pth')
    model_checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(model_checkpoint['model_state_dict'])

    start_time = time()
    test_loss, test_metrics = valid(accelerator, dataloaders_dict["test"],
                                                        tools)
    end_time = time()
    if accelerator.is_main_process:
        logging.info(
            f'\nbest_model_loss'
            f'\ntest_family - time {np.round(end_time - start_time, 2)}s, '
            f'test loss {np.round(test_loss, 4)}')
        logging.info(f'test auc: {np.round(test_metrics[0], 4)}')
        logging.info(f'test aps: {np.round(test_metrics[1], 4)}')
        logging.info(f'test max f1: {np.round(test_metrics[2], 4)}')
        # logging.info(f'test macro f1: {np.round(test_macro_f1, 4)}')
        # test_f1 = [np.round(v, 4) for v in test_f1.tolist()]
        # logging.info(f'test f1: {test_f1}')

    model_path = os.path.join(tools['result_path'], 'checkpoints', 'best_model_auc.pth')
    model_checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(model_checkpoint['model_state_dict'])

    start_time = time()
    test_loss, test_metrics = valid(accelerator, dataloaders_dict["test"],
                                                        tools)
    end_time = time()
    if accelerator.is_main_process:
        logging.info(
            f'\nbest_model_auc'
            f'\ntest_super_family - time {np.round(end_time - start_time, 2)}s, '
            f'test loss {np.round(test_loss, 4)}')
        logging.info(f'test auc: {np.round(test_metrics[0], 4)}')
        logging.info(f'test aps: {np.round(test_metrics[1], 4)}')
        logging.info(f'test max f1: {np.round(test_metrics[2], 4)}')
        # logging.info(f'test macro f1: {np.round(test_macro_f1, 4)}')
        # test_f1 = [np.round(v, 4) for v in test_f1.tolist()]
        # logging.info(f'test f1: {test_f1}')

    # start_time = time()
    # test_loss, test_acc, test_macro_f1, test_f1 = valid(0, accelerator, dataloaders_dict["test_fold"],
    #                                                     tools, tensorboard_log=False)
    # end_time = time()
    # if accelerator.is_main_process:
    #     logging.info(
    #         f'\ntest_fold - time {np.round(end_time - start_time, 2)}s, '
    #         f'test loss {np.round(test_loss, 4)}')
    #     logging.info(f'test acc: {np.round(test_acc, 4)}')
    #     logging.info(f'test macro f1: {np.round(test_macro_f1, 4)}')
        # test_f1 = [np.round(v, 4) for v in test_f1.tolist()]
        # logging.info(f'test f1: {test_f1}')

    accelerator.end_training()
    accelerator.free_memory()
    del tools, net, dataloaders_dict, accelerator, optimizer, scheduler
    torch.cuda.empty_cache()


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
    print('done!')