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
from data.data_ddt import prepare_dataloaders_fold, prepare_dataloaders_from_dms
from model_ddt import prepare_models
from accelerate import Accelerator
import torch.nn.functional as F
from scipy.stats import spearmanr
# from focal_loss import FocalLoss


def train(epoch, accelerator, dataloader, tools, global_step, tensorboard_log, configs):
    tools["net"].train()
    tools["optimizer"].zero_grad()

    # Track mean loss per optimizer (sync_gradients) step
    step_losses = []
    preds_all = []
    labels_all = []

    progress_bar = tqdm(
        range(global_step, global_step + int(np.ceil(len(dataloader) / tools["accum_iter"]))),
        disable=not accelerator.is_local_main_process,
        leave=False,
    )
    progress_bar.set_description("Steps")

    for i, data in enumerate(dataloader):
        with accelerator.accumulate(tools["net"]):
            wt, mut, ddg, prot_ids = data
            outputs = tools["net"](wt, mut)

            # Use per-sample losses (important if you later want weights)
            if tools["loss_function"] == F.mse_loss:
                losses = F.mse_loss(outputs, ddg, reduction="none")
            else:
                # if you later add other losses
                losses = tools["loss_function"](outputs, ddg)

            loss = losses.mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(tools["net"].parameters(), tools["grad_clip"])
                tools["optimizer"].step()
                tools["scheduler"].step()
                tools["optimizer"].zero_grad()

        # Only log/accumulate once per optimizer step
        if accelerator.sync_gradients:
            bs = ddg.shape[0]
            avg_loss = accelerator.gather(loss.detach().repeat(bs)).mean().item()
            step_losses.append(avg_loss)

            # Gather predictions/labels for metrics (flatten to 1D)
            gathered_preds = accelerator.gather(outputs.detach()).float().cpu()  # [global_B]
            gathered_labels = accelerator.gather(ddg.detach()).float().cpu()     # [global_B]
            preds_all.append(gathered_preds)
            labels_all.append(gathered_labels)

            if tensorboard_log and accelerator.is_main_process:
                tools["train_writer"].add_scalar("step_loss", avg_loss, global_step)
                tools["train_writer"].add_scalar("learning_rate", tools["optimizer"].param_groups[0]["lr"], global_step)

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                step_loss=avg_loss,
                lr=tools["optimizer"].param_groups[0]["lr"],
            )

    # ---- epoch metrics ----
    epoch_loss = float(np.mean(step_losses)) if len(step_losses) > 0 else 0.0

    epoch_spearman_cor = 0.0
    if len(preds_all) > 0:
        preds = torch.cat(preds_all).numpy()
        labels = torch.cat(labels_all).numpy()

        # compute Spearman only on main process to avoid duplicate logs
        if accelerator.is_main_process:
            epoch_spearman_cor, _ = spearmanr(preds, labels)
            accelerator.log({"train_epoch_spearman_cor": epoch_spearman_cor}, step=epoch)
            if tensorboard_log:
                tools["train_writer"].add_scalar("spearman_cor", epoch_spearman_cor, epoch)

    return epoch_loss, epoch_spearman_cor, global_step



def valid(epoch, accelerator, dataloader, tools, tensorboard_log):
    tools["net"].eval()

    step_losses = []
    preds_all = []
    labels_all = []

    progress_bar = tqdm(
        range(len(dataloader)),
        disable=not accelerator.is_local_main_process,
        leave=False,
    )
    progress_bar.set_description("Steps")

    for i, data in enumerate(dataloader):
        wt, mut, ddg, prot_ids = data

        with torch.inference_mode():
            outputs = tools["net"](wt, mut)

            if tools["loss_function"] == F.mse_loss:
                losses = F.mse_loss(outputs, ddg, reduction="none")
            else:
                losses = tools["loss_function"](outputs, ddg)

            loss = losses.mean()

        bs = ddg.shape[0]
        avg_loss = accelerator.gather(loss.detach().repeat(bs)).mean().item()
        step_losses.append(avg_loss)

        gathered_preds = accelerator.gather(outputs.detach()).float().cpu()
        gathered_labels = accelerator.gather(ddg.detach()).float().cpu()
        preds_all.append(gathered_preds)
        labels_all.append(gathered_labels)

        progress_bar.update(1)
        progress_bar.set_postfix(
            step_loss=avg_loss,
            lr=tools["optimizer"].param_groups[0]["lr"],
        )

    valid_loss = float(np.mean(step_losses)) if len(step_losses) > 0 else 0.0

    valid_spearman = 0.0
    if len(preds_all) > 0:
        preds = torch.cat(preds_all).numpy()
        labels = torch.cat(labels_all).numpy()

        if accelerator.is_main_process:
            valid_spearman, _ = spearmanr(preds, labels)
            accelerator.log({"valid_cor": valid_spearman}, step=epoch)
            if tensorboard_log:
                tools["valid_writer"].add_scalar("cor", valid_spearman, epoch)

    return valid_loss, valid_spearman



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
    dataloaders_dict = prepare_dataloaders_fold(configs)
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
    dataloaders_dict = prepare_dataloaders_from_dms(configs)
    dataloaders_dict["train"], dataloaders_dict["valid"] = accelerator.prepare(
        dataloaders_dict["train"],
        dataloaders_dict["valid"]
    )
    best_valid_mse = np.inf
    best_valid_spearman = 0
    global_step = 0
    for epoch in range(0, 2*configs.train_settings.num_epochs + 1):
        tools['epoch'] = epoch
        start_time = time()
        train_loss, train_cor, global_step = train(epoch, accelerator, dataloaders_dict["train"],
                                                tools, global_step,
                                                configs.tensorboard_log, configs)
        end_time = time()
        if accelerator.is_main_process:
            logging.info(f'epoch {epoch} - time {np.round(end_time - start_time, 2)}s, '
                         f'train loss {np.round(train_loss, 4)}, train cor {np.round(train_cor, 4)}')

        if epoch % configs.valid_settings.do_every == 0 and epoch != 0:
            start_time = time()
            valid_loss, valid_cor = valid(epoch, accelerator,
                                          dataloaders_dict["valid"],
                                          tools,
                                          tensorboard_log=False)
            end_time = time()
            if accelerator.is_main_process:
                logging.info(
                    f'evaluation - time {np.round(end_time - start_time, 2)}s, '
                    f'valid loss {np.round(valid_loss, 6)}, valid cor {np.round(valid_cor, 6)}')

            if valid_loss< best_valid_mse:
                best_valid_mse = valid_loss
                model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_dms_model_mse.pth')
                accelerator.wait_for_everyone()
                save_checkpoint_infer(epoch, model_path, tools, accelerator)
            if valid_cor > best_valid_spearman:
                best_valid_spearman = valid_cor
                model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_dms_model_cor.pth')
                save_checkpoint_infer(epoch, model_path, tools, accelerator)
            

        # if epoch % configs.checkpoints_every == 0:
        #     # Set the path to save the model checkpoint.
        #     model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')
        #     accelerator.wait_for_everyone()
        #     save_checkpoint(epoch, model_path, tools, accelerator)

    if accelerator.is_main_process:
        logging.info(f'best valid mse: {np.round(best_valid_mse, 6)}')
        logging.info(f'best valid cor: {np.round(best_valid_spearman, 6)}')

    sleep(20)

    model_path = os.path.join(tools['result_path'], 'checkpoints', 'best_dms_model_mse.pth')
    model_checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(model_checkpoint['model_state_dict'])

    optimizer, scheduler = prepare_optimizer_infer(net, configs, len(dataloaders_dict["train"]), logging)
    net, optimizer, scheduler = accelerator.prepare(net, optimizer, scheduler)
    tools["optimizer"] = optimizer
    tools["scheduler"] = scheduler


    dataloaders_dict = prepare_dataloaders_fold(configs)
    dataloaders_dict["train"], dataloaders_dict["valid"], dataloaders_dict['test'] = accelerator.prepare(
        dataloaders_dict["train"],
        dataloaders_dict["valid"],
        dataloaders_dict['test']
    )
    best_valid_mse = np.inf
    best_valid_spearman = 0
    global_step = 0
    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
        tools['epoch'] = epoch
        start_time = time()
        train_loss, train_cor, global_step = train(epoch, accelerator, dataloaders_dict["train"],
                                                tools, global_step,
                                                configs.tensorboard_log, configs)
        end_time = time()
        if accelerator.is_main_process:
            logging.info(f'epoch {epoch} - time {np.round(end_time - start_time, 2)}s, '
                         f'train loss {np.round(train_loss, 4)}, train cor {np.round(train_cor, 4)}')

        if epoch % configs.valid_settings.do_every == 0 and epoch != 0:
            start_time = time()
            valid_loss, valid_cor = valid(epoch, accelerator,
                                          dataloaders_dict["valid"],
                                          tools,
                                          tensorboard_log=False)
            end_time = time()
            if accelerator.is_main_process:
                logging.info(
                    f'evaluation - time {np.round(end_time - start_time, 2)}s, '
                    f'valid loss {np.round(valid_loss, 6)}, valid cor {np.round(valid_cor, 6)}')

            if valid_loss< best_valid_mse:
                best_valid_mse = valid_loss
                model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model_mse.pth')
                accelerator.wait_for_everyone()
                save_checkpoint_infer(epoch, model_path, tools, accelerator)
            if valid_cor > best_valid_spearman:
                best_valid_spearman = valid_cor
                model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model_cor.pth')
                save_checkpoint_infer(epoch, model_path, tools, accelerator)
            

        # if epoch % configs.checkpoints_every == 0:
        #     # Set the path to save the model checkpoint.
        #     model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')
        #     accelerator.wait_for_everyone()
        #     save_checkpoint(epoch, model_path, tools, accelerator)

    if accelerator.is_main_process:
        logging.info(f'best valid mse: {np.round(best_valid_mse, 6)}')
        logging.info(f'best valid cor: {np.round(best_valid_spearman, 6)}')
        # best_valid_f1 = [np.round(v, 4) for v in best_valid_f1.tolist()]
        # logging.info(f'best valid f1: {best_valid_f1}')

    train_writer.close()
    valid_writer.close()

    # pause 20 second to make sure the best validation checkpoint is ready on the disk
    sleep(20)

    model_path = os.path.join(tools['result_path'], 'checkpoints', 'best_model_mse.pth')
    model_checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(model_checkpoint['model_state_dict'])

    start_time = time()
    test_loss, test_cor = valid(0, accelerator, dataloaders_dict["test"],
                                                        tools, tensorboard_log=False)
    end_time = time()
    if accelerator.is_main_process:
        logging.info(
            f'\nbest_model_mse'
            f'\ntest_family - time {np.round(end_time - start_time, 2)}s, '
            f'test loss {np.round(test_loss, 4)}')
        logging.info(f'test cor: {np.round(test_cor, 4)}')
        # logging.info(f'test macro f1: {np.round(test_macro_f1, 4)}')
        # test_f1 = [np.round(v, 4) for v in test_f1.tolist()]
        # logging.info(f'test f1: {test_f1}')

    model_path = os.path.join(tools['result_path'], 'checkpoints', 'best_model_cor.pth')
    model_checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(model_checkpoint['model_state_dict'])

    start_time = time()
    test_loss, test_cor = valid(0, accelerator, dataloaders_dict["test"],
                                                        tools, tensorboard_log=False)
    end_time = time()
    if accelerator.is_main_process:
        logging.info(
            f'\nbest_model_cor'
            f'\ntest_super_family - time {np.round(end_time - start_time, 2)}s, '
            f'test loss {np.round(test_loss, 4)}')
        logging.info(f'test cor: {np.round(test_cor, 4)}')
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