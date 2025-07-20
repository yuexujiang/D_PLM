import argparse
import os
import yaml
import time
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from model import prepare_models, AverageMeter, info_nce_loss, info_nce_weights
from model import log_negative_mean_logtis
from model import MaskedLMDataCollator

from utils.utils import prepare_optimizer, load_checkpoints, load_checkpoints_md, save_checkpoints, load_configs, test_gpu_cuda, prepare_saving_dir, save_best_checkpoints
from utils.utils import get_logging, prepare_tensorboard
from utils.utils import accuracy, residue_batch_sample
from utils.evaluation import evaluate_with_deaminase, evaluate_with_kinase, evaluate_with_cath_more, \
                             evaluate_with_deaminase_md, evaluate_with_kinase_md, evaluate_with_cath_more_md
from utils.cath_with_struct import evaluate_with_cath_more_struct


#def prepare_loss(features_struct, features_seq, residue_struct, residue_seq, plddt_residue, criterion, loss,
#                 accelerator, configs):
def prepare_loss(simclr,graph,batch_tokens,plddt,criterion, loss,
                 accelerator, configs,
                 masked_lm_data_collator
                 ):
    
    #simclr_loss, MLM_loss,simclr_residue_loss, logits, labels, logits_residue, labels_residue = 0,0,0,0,0,0,0
    MLM_loss=0
    features_struct, residue_struct, features_seq, residue_seq = simclr(
                    graph=graph, batch_tokens=batch_tokens)
    
    if plddt == None:
        residue_struct, residue_seq = residue_batch_sample(
                    residue_struct, residue_seq, plddt,
                    configs.train_settings.residue_batch_size, accelerator)
    else:
        residue_struct, residue_seq, plddt_residue = residue_batch_sample(
                    residue_struct, residue_seq, plddt,
                    configs.train_settings.residue_batch_size, accelerator)
    
    logits, labels = info_nce_loss(features_struct, features_seq, configs.train_settings.n_views,
                               configs.train_settings.temperature, accelerator)
    logits_residue, labels_residue = info_nce_loss(residue_struct, residue_seq, configs.train_settings.n_views,
                                                   configs.train_settings.temperature, accelerator)
    
    if configs.train_settings.plddt_weight:
        # simclr_residue_loss = torch.mean(
        #    criterion(logits_residue, labels_residue) * torch.cat([plddt_residue, plddt_residue],
        #                                                          dim=0))
        seq_weight = torch.ones(plddt_residue.shape)
        avg_plddt_weights = info_nce_weights(plddt_residue, seq_weight, device=accelerator.device,
                                             mode=configs.train_settings.plddt_weight_mode)
        simclr_residue_loss = torch.mean(criterion(logits_residue * avg_plddt_weights, labels_residue))
    else:
        simclr_residue_loss = torch.mean(criterion(logits_residue, labels_residue))
    
    simclr_loss = torch.mean(criterion(logits, labels))
    loss += simclr_loss + simclr_residue_loss
    if hasattr(configs.model.esm_encoder,"MLM"):
        if configs.model.esm_encoder.MLM.enable:
            # make masked input and label
            mlm_inputs, mlm_labels = masked_lm_data_collator.mask_tokens(batch_tokens)
            #print(mlm_inputs) have tested
            if hasattr(configs.model.esm_encoder.MLM,"mode") and configs.model.esm_encoder.MLM.mode=="contrast":
                features_seq_mask,_ = simclr.model_seq(mlm_inputs)
                logits_mask, labels_mask = info_nce_loss(features_seq_mask, features_seq,configs.train_settings.n_views,
                                                   configs.train_settings.temperature, accelerator)
                MLM_loss = torch.mean(criterion(logits_mask, labels_mask))
            else:
               prediction_scores = simclr.model_seq(mlm_inputs,return_logits=True)# [seqlen,33]
               # CrossEntropyLoss
               vocab_size = simclr.model_seq.alphabet.all_toks.__len__()
               MLM_loss = torch.mean(criterion(prediction_scores.view(-1, vocab_size), mlm_labels.view(-1)))
            
            loss += MLM_loss
    
    return loss, simclr_loss,simclr_residue_loss,MLM_loss, logits, labels, logits_residue, labels_residue


def training_loop(simclr, start_step,train_loader, val_loader, batch_converter, criterion,
                  optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
                  result_path, logging, configs, masked_lm_data_collator=None, **kwargs):
    torch.cuda.empty_cache()
    accelerator = kwargs['accelerator']

    train_loss = 0
    n_steps = start_step
    n_sub_steps = start_step-1
    epoch_num = 0
    """
    train_alt = False
    if hasattr(configs.model.esm_encoder, "MLM"):
        if configs.model.esm_encoder.MLM.enable and configs.model.esm_encoder.MLM.alt:
            train_alt = True
    """
    
    """
    if accelerator.is_main_process:
       seq_evaluation_loop(simclr, n_steps, configs, batch_converter, result_path, valid_writer,logging, accelerator)
       struct_evaluation_loop(simclr, n_steps, configs, result_path, valid_writer,logging, accelerator)
    """
    while True:
        epoch_num += 1
        if accelerator.is_main_process:
            logging.info(f"Epoch {epoch_num}")

        losses = AverageMeter()
        
        progress_bar = tqdm(range(0, int(np.ceil(len(train_loader) / configs.train_settings.gradient_accumulation))),
                            disable=True, leave=True, desc=f"Epoch {epoch_num} steps")
        bsz = configs.train_settings.batch_size
        start = time.time()
        for idx, batch in enumerate(train_loader):
            with accelerator.accumulate(simclr):
                simclr.train()
                batch_seq = [(batch['pid'][i], str(batch['seq'][i])) for i in range(len(batch['seq']))]  # batch['seq']
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
                graph = batch["graph"]
                batch_tokens = batch_tokens.to(accelerator.device)
                graph = graph.to(accelerator.device)
                loss = torch.tensor(0).float()
                loss = loss.to(accelerator.device)
                loss, simclr_loss, simclr_residue_loss, MLM_loss,logits, labels, logits_residue, labels_residue = prepare_loss(
                        simclr,graph,batch_tokens,batch['plddt'],criterion,loss, accelerator, configs,
                        masked_lm_data_collator
                        )
                    
                """
                if (train_alt and n_sub_steps % 2 == 0) or not train_alt:
                    features_struct, residue_struct, features_seq, residue_seq = simclr(
                        graph=graph, batch_tokens=batch_tokens)

                    residue_struct, residue_seq, plddt_residue = residue_batch_sample(
                        residue_struct, residue_seq, batch['plddt'],
                        configs.train_settings.residue_batch_size, accelerator)

                    loss, simclr_loss, simclr_residue_loss, logits, labels, logits_residue, labels_residue = prepare_loss(
                        features_struct, features_seq, residue_struct, residue_seq, plddt_residue, criterion,
                        loss, accelerator, configs,
                        )
                    """

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(configs.train_settings.batch_size)).mean()
                train_loss += avg_loss.item() / configs.train_settings.gradient_accumulation

                accelerator.backward(loss)

                # Step the optimizers
                if not (configs.model.struct_encoder.fine_tuning.enable is False and configs.model.struct_encoder.fine_tuning_projct.enable is False):
                    optimizer_x.step()
                
                optimizer_seq.step()

                # Step the schedulers
                if not (configs.model.struct_encoder.fine_tuning.enable is False and configs.model.struct_encoder.fine_tuning_projct.enable is False):
                    scheduler_x.step()
                
                scheduler_seq.step()

                losses.update(loss.item(), bsz)

                # Zero the parameter gradients
                if not (configs.model.struct_encoder.fine_tuning.enable is False and configs.model.struct_encoder.fine_tuning_projct.enable is False):
                    optimizer_x.zero_grad()
                optimizer_seq.zero_grad()

            if accelerator.sync_gradients:
                train_writer.add_scalar('step loss', train_loss, n_steps)
                train_writer.add_scalar('learning rate (structure)', optimizer_x.param_groups[0]['lr'], n_steps)
                train_writer.add_scalar('learning rate (sequence)', optimizer_seq.param_groups[0]['lr'], n_steps)
                # writer.add_scalar('loss', loss, global_step=n_steps)
                # writer.add_scalar('acc/top1', top1[0], global_step=n_steps)

                if n_steps % configs.checkpoints_every == 0 and n_steps != 0:
                    # Wait for other processes to catch up
                    print("save_checkpoints")
                    accelerator.wait_for_everyone()
                    # Save checkpoint
                    save_checkpoints(accelerator.unwrap_model(optimizer_x),
                                     accelerator.unwrap_model(optimizer_seq),
                                     result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num)

                # todo: add tensorboard here
                if n_steps % configs.valid_settings.do_every == 0 and n_steps != 0:  # I want to see the step evaluations
                    print("in evaluation loop")
                    # Protein level
                    accelerator.wait_for_everyone()
                    evaluation_loop(simclr, val_loader, labels, labels_residue, batch_converter, criterion, configs,
                                    simclr_loss=simclr_loss, simclr_residue_loss=simclr_residue_loss, MLM_loss = MLM_loss,losses=losses,
                                    n_steps=n_steps, scheduler_seq=scheduler_seq, result_path=result_path,
                                    scheduler_struct=scheduler_x, logits=logits, logits_residue=logits_residue,
                                    loss=loss, bsz=bsz, train_writer=train_writer,valid_writer=valid_writer,
                                    masked_lm_data_collator=masked_lm_data_collator, logging=logging,
                                    accelerator=accelerator)

                if configs.valid_settings.eval_struct.enable and (
                        n_steps % configs.valid_settings.eval_struct.do_every == 0) and n_steps != 0:  # or n_steps == 1):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                       struct_evaluation_loop(simclr, n_steps, configs, result_path, valid_writer,logging, accelerator)

                progress_bar.update(1)
                n_steps += 1
                train_loss = 0

            n_sub_steps += 1
            if n_steps > configs.train_settings.num_steps:
                break

        end = time.time()

        if n_steps > configs.train_settings.num_steps:
            break

        if accelerator.is_main_process:
            logging.info(f"one epoch cost {(end - start):.2f}, number of trained steps {n_steps}")



def training_loop_DCCM_GNN(simclr, start_step, train_loader, val_loader, test_loader, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, replicate, masked_lm_data_collator=None, **kwargs):
    torch.cuda.empty_cache()
    accelerator = kwargs['accelerator']

    train_loss = 0
    n_steps = start_step
    n_sub_steps = start_step-1
    epoch_num = 0
    """
    train_alt = False
    if hasattr(configs.model.esm_encoder, "MLM"):
        if configs.model.esm_encoder.MLM.enable and configs.model.esm_encoder.MLM.alt:
            train_alt = True
    """
    
    """
    if accelerator.is_main_process:
       seq_evaluation_loop(simclr, n_steps, configs, batch_converter, result_path, valid_writer,logging, accelerator)
       struct_evaluation_loop(simclr, n_steps, configs, result_path, valid_writer,logging, accelerator)
    """
    while True:
        epoch_num += 1
        if accelerator.is_main_process:
            logging.info(f"Epoch {epoch_num}")

        losses = AverageMeter()
        
        progress_bar = tqdm(range(0, int(np.ceil(len(train_loader) / configs.train_settings.gradient_accumulation))),
                            disable=True, leave=True, desc=f"Epoch {epoch_num} steps")
        bsz = configs.train_settings.batch_size
        start = time.time()
        for idx, batch in enumerate(train_loader):
            with accelerator.accumulate(simclr):
                simclr.train()
                batch_seq = [(batch['pid'][i], str(batch['seq'][i])) for i in range(len(batch['seq']))]  # batch['seq']
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
                graph = batch["graph"]
                # trajs = batch["traj"]
                batch_tokens = batch_tokens.to(accelerator.device)
                graph = graph.to(accelerator.device)
                # trajs = trajs.to(accelerator.device)
                # trajs = torch.tensor(np.array(trajs)).to(accelerator.device)
                loss = torch.tensor(0).float()
                loss = loss.to(accelerator.device)
                loss, simclr_loss, simclr_residue_loss, MLM_loss,logits, labels, logits_residue, labels_residue= prepare_loss(
                        simclr,graph,batch_tokens, None, criterion,loss, accelerator, configs,
                        masked_lm_data_collator
                        )
                    

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(configs.train_settings.batch_size)).mean()
                train_loss += avg_loss.item() / configs.train_settings.gradient_accumulation

                accelerator.backward(loss)

                # Step the optimizers
                if not (configs.model.DCCM_GNN_encoder.fine_tuning.enable is False and configs.model.DCCM_GNN_encoder.fine_tuning_projct.enable is False):
                    optimizer_x.step()
                
                optimizer_seq.step()

                # Step the schedulers
                if not (configs.model.DCCM_GNN_encoder.fine_tuning.enable is False and configs.model.DCCM_GNN_encoder.fine_tuning_projct.enable is False):
                    scheduler_x.step()
                
                scheduler_seq.step()

                losses.update(loss.item(), bsz)

                # Zero the parameter gradients
                if not (configs.model.DCCM_GNN_encoder.fine_tuning.enable is False and configs.model.DCCM_GNN_encoder.fine_tuning_projct.enable is False):
                    optimizer_x.zero_grad()
                optimizer_seq.zero_grad()

            if accelerator.sync_gradients:
                train_writer.add_scalar('step loss', train_loss, n_steps)
                train_writer.add_scalar('learning rate (MD)', optimizer_x.param_groups[0]['lr'], n_steps)
                train_writer.add_scalar('learning rate (sequence)', optimizer_seq.param_groups[0]['lr'], n_steps)
                # writer.add_scalar('loss', loss, global_step=n_steps)
                # writer.add_scalar('acc/top1', top1[0], global_step=n_steps)

                if n_steps % configs.checkpoints_every == 0 and n_steps != 0:
                    # Wait for other processes to catch up
                    print("save_checkpoints")
                    accelerator.wait_for_everyone()
                    # Save checkpoint
                    save_checkpoints(accelerator.unwrap_model(optimizer_x),
                                     accelerator.unwrap_model(optimizer_seq),
                                     result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num)

                # todo: add tensorboard here
                if n_steps % configs.valid_settings.do_every == 0 and n_steps != 0:  # I want to see the step evaluations
                    print("in evaluation loop")
                    # Protein level
                    accelerator.wait_for_everyone()
                    # evaluation_loop_MD(simclr, val_loader, labels, batch_converter, criterion, configs,
                    #                 simclr_loss=simclr_loss, MLM_loss = MLM_loss,losses=losses,
                    #                 n_steps=n_steps, scheduler_seq=scheduler_seq, result_path=result_path,
                    #                 scheduler_x=scheduler_x, logits=logits, 
                    #                 loss=loss, bsz=bsz, train_writer=train_writer,valid_writer=valid_writer,
                    #                 masked_lm_data_collator=masked_lm_data_collator, logging=logging,
                    #                 accelerator=accelerator)
                    evaluation_loop(simclr, val_loader, labels, labels_residue, batch_converter, criterion, configs,
                                    simclr_loss=simclr_loss, simclr_residue_loss=simclr_residue_loss, MLM_loss = MLM_loss,losses=losses,
                                    n_steps=n_steps, scheduler_seq=scheduler_seq, result_path=result_path,
                                    scheduler_struct=scheduler_x, logits=logits, logits_residue=logits_residue,
                                    loss=loss, bsz=bsz, train_writer=train_writer,valid_writer=valid_writer,
                                    masked_lm_data_collator=masked_lm_data_collator, logging=logging,
                                    accelerator=accelerator)

                # if configs.valid_settings.eval_struct.enable and (
                #         n_steps % configs.valid_settings.eval_struct.do_every == 0) and n_steps != 0:  # or n_steps == 1):
                #     accelerator.wait_for_everyone()
                #     if accelerator.is_main_process:
                #        struct_evaluation_loop(simclr, n_steps, configs, result_path, valid_writer,logging, accelerator)

                progress_bar.update(1)
                n_steps += 1
                train_loss = 0

            n_sub_steps += 1
            repli_num = configs.train_settings.num_steps if replicate == 2 else (configs.train_settings.num_steps // 3) * (replicate + 1)
            if n_steps > repli_num:
                break

        end = time.time()

        repli_num = configs.train_settings.num_steps if replicate == 2 else (configs.train_settings.num_steps // 3) * (replicate + 1)
        if n_steps > repli_num:
            break

        if accelerator.is_main_process:
            logging.info(f"one epoch cost {(end - start):.2f}, number of trained steps {n_steps}")

    return n_steps, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq

def training_loop_Geom2ve_tematt(simclr, start_step, train_loader, val_loader, test_loader, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, replicate, masked_lm_data_collator=None, **kwargs):
    torch.cuda.empty_cache()
    accelerator = kwargs['accelerator']

    train_loss = 0
    n_steps = start_step
    n_sub_steps = start_step-1
    epoch_num = 0
    """
    train_alt = False
    if hasattr(configs.model.esm_encoder, "MLM"):
        if configs.model.esm_encoder.MLM.enable and configs.model.esm_encoder.MLM.alt:
            train_alt = True
    """
    
    """
    if accelerator.is_main_process:
       seq_evaluation_loop(simclr, n_steps, configs, batch_converter, result_path, valid_writer,logging, accelerator)
       struct_evaluation_loop(simclr, n_steps, configs, result_path, valid_writer,logging, accelerator)
    """
    best_val_loss = np.inf
    best_val_graph_loss = np.inf
    best_val_residue_loss = np.inf
    while True:
        epoch_num += 1
        if accelerator.is_main_process:
            logging.info(f"Epoch {epoch_num}")

        losses = AverageMeter()
        
        progress_bar = tqdm(range(0, int(np.ceil(len(train_loader) / configs.train_settings.gradient_accumulation))),
                            disable=True, leave=True, desc=f"Epoch {epoch_num} steps")
        bsz = configs.train_settings.batch_size
        start = time.time()
        for idx, batch in enumerate(train_loader):
            with accelerator.accumulate(simclr):
                simclr.train()
                batch_seq = [(batch['pid'][i], str(batch['seq'][i])) for i in range(len(batch['seq']))]  # batch['seq']
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
                # graph = batch["graph"]
                res_rep = batch['res_rep']
                batch_idx = batch['batch_idx']
                graph = [res_rep.to(accelerator.device), batch_idx.to(accelerator.device)]
                # trajs = batch["traj"]
                batch_tokens = batch_tokens.to(accelerator.device)
                # graph = graph.to(accelerator.device)
                # trajs = trajs.to(accelerator.device)
                # trajs = torch.tensor(np.array(trajs)).to(accelerator.device)
                loss = torch.tensor(0).float()
                loss = loss.to(accelerator.device)
                loss, simclr_loss, simclr_residue_loss, MLM_loss,logits, labels, logits_residue, labels_residue= prepare_loss(
                        simclr,graph,batch_tokens, None, criterion,loss, accelerator, configs,
                        masked_lm_data_collator
                        )
                    

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(configs.train_settings.batch_size)).mean()
                train_loss += avg_loss.item() / configs.train_settings.gradient_accumulation

                accelerator.backward(loss)

                # Step the optimizers
                if not (configs.model.DCCM_GNN_encoder.fine_tuning.enable is False and configs.model.DCCM_GNN_encoder.fine_tuning_projct.enable is False):
                    optimizer_x.step()
                
                optimizer_seq.step()

                # Step the schedulers
                if not (configs.model.DCCM_GNN_encoder.fine_tuning.enable is False and configs.model.DCCM_GNN_encoder.fine_tuning_projct.enable is False):
                    scheduler_x.step()
                
                scheduler_seq.step()

                losses.update(loss.item(), bsz)

                # Zero the parameter gradients
                if not (configs.model.DCCM_GNN_encoder.fine_tuning.enable is False and configs.model.DCCM_GNN_encoder.fine_tuning_projct.enable is False):
                    optimizer_x.zero_grad()
                optimizer_seq.zero_grad()

            if accelerator.sync_gradients:
                train_writer.add_scalar('step loss', train_loss, n_steps)
                train_writer.add_scalar('learning rate (MD)', optimizer_x.param_groups[0]['lr'], n_steps)
                train_writer.add_scalar('learning rate (sequence)', optimizer_seq.param_groups[0]['lr'], n_steps)
                # writer.add_scalar('loss', loss, global_step=n_steps)
                # writer.add_scalar('acc/top1', top1[0], global_step=n_steps)

                if n_steps % configs.checkpoints_every == 0 and n_steps != 0:
                    # Wait for other processes to catch up
                    print("save_checkpoints")
                    accelerator.wait_for_everyone()
                    # Save checkpoint
                    save_checkpoints(accelerator.unwrap_model(optimizer_x),
                                     accelerator.unwrap_model(optimizer_seq),
                                     result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num)

                # todo: add tensorboard here
                if n_steps % configs.valid_settings.do_every == 0 and n_steps != 0:  # I want to see the step evaluations
                    print("in evaluation loop")
                    # Protein level
                    accelerator.wait_for_everyone()
                    # evaluation_loop_MD(simclr, val_loader, labels, batch_converter, criterion, configs,
                    #                 simclr_loss=simclr_loss, MLM_loss = MLM_loss,losses=losses,
                    #                 n_steps=n_steps, scheduler_seq=scheduler_seq, result_path=result_path,
                    #                 scheduler_x=scheduler_x, logits=logits, 
                    #                 loss=loss, bsz=bsz, train_writer=train_writer,valid_writer=valid_writer,
                    #                 masked_lm_data_collator=masked_lm_data_collator, logging=logging,
                    #                 accelerator=accelerator)
                    loss_val, val_graph_loss, val_residue_loss = evaluation_loop(simclr, val_loader, labels, labels_residue, batch_converter, criterion, configs,
                                    simclr_loss=simclr_loss, simclr_residue_loss=simclr_residue_loss, MLM_loss = MLM_loss,losses=losses,
                                    n_steps=n_steps, scheduler_seq=scheduler_seq, result_path=result_path,
                                    scheduler_struct=scheduler_x, logits=logits, logits_residue=logits_residue,
                                    loss=loss, bsz=bsz, train_writer=train_writer,valid_writer=valid_writer,
                                    masked_lm_data_collator=masked_lm_data_collator, logging=logging,
                                    accelerator=accelerator)
                    best_val_loss = save_best_checkpoints(accelerator.unwrap_model(optimizer_x),
                                     accelerator.unwrap_model(optimizer_seq),
                                     result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num,
                                     best_val_loss, loss_val, 'best_val_whole_loss')
                    best_val_graph_loss = save_best_checkpoints(accelerator.unwrap_model(optimizer_x),
                                     accelerator.unwrap_model(optimizer_seq),
                                     result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num,
                                     best_val_graph_loss, val_graph_loss, 'best_val_graph_loss')
                    best_val_residue_loss = save_best_checkpoints(accelerator.unwrap_model(optimizer_x),
                                     accelerator.unwrap_model(optimizer_seq),
                                     result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num,
                                     best_val_residue_loss, val_residue_loss, 'best_val_residue_loss')

                # if configs.valid_settings.eval_struct.enable and (
                #         n_steps % configs.valid_settings.eval_struct.do_every == 0) and n_steps != 0:  # or n_steps == 1):
                #     accelerator.wait_for_everyone()
                #     if accelerator.is_main_process:
                #        struct_evaluation_loop(simclr, n_steps, configs, result_path, valid_writer,logging, accelerator)

                progress_bar.update(1)
                n_steps += 1
                train_loss = 0

            n_sub_steps += 1
            repli_num = configs.train_settings.num_steps if replicate == 2 else (configs.train_settings.num_steps // 3) * (replicate + 1)
            if n_steps > repli_num:
            # if n_steps > configs.train_settings.num_steps:
                break

        end = time.time()

        repli_num = configs.train_settings.num_steps if replicate == 2 else (configs.train_settings.num_steps // 3) * (replicate + 1)
        if n_steps > repli_num:
        # if n_steps > configs.train_settings.num_steps:
            break

        if accelerator.is_main_process:
            logging.info(f"one epoch cost {(end - start):.2f}, number of trained steps {n_steps}")

    return n_steps, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq

def prepare_loss_MD(simclr,traj,batch_tokens,criterion, loss,
                 accelerator, configs,
                 masked_lm_data_collator
                 ):
    
    #simclr_loss, MLM_loss,simclr_residue_loss, logits, labels, logits_residue, labels_residue = 0,0,0,0,0,0,0
    MLM_loss=0
    features_MD, features_seq, residue_seq = simclr(
                    graph=traj, batch_tokens=batch_tokens)
    
    # residue_struct, residue_seq, plddt_residue = residue_batch_sample(
    #                 residue_struct, residue_seq, plddt,
    #                 configs.train_settings.residue_batch_size, accelerator)
    
    logits, labels = info_nce_loss(features_MD, features_seq, configs.train_settings.n_views,
                               configs.train_settings.temperature, accelerator)
    # logits_residue, labels_residue = info_nce_loss(residue_struct, residue_seq, configs.train_settings.n_views,
    #                                                configs.train_settings.temperature, accelerator)
    
    # if configs.train_settings.plddt_weight:
    #     # simclr_residue_loss = torch.mean(
    #     #    criterion(logits_residue, labels_residue) * torch.cat([plddt_residue, plddt_residue],
    #     #                                                          dim=0))
    #     seq_weight = torch.ones(plddt_residue.shape)
    #     avg_plddt_weights = info_nce_weights(plddt_residue, seq_weight, device=accelerator.device,
    #                                          mode=configs.train_settings.plddt_weight_mode)
    #     simclr_residue_loss = torch.mean(criterion(logits_residue * avg_plddt_weights, labels_residue))
    # else:
    #     simclr_residue_loss = torch.mean(criterion(logits_residue, labels_residue))
    
    simclr_loss = torch.mean(criterion(logits, labels))
    # loss += simclr_loss + simclr_residue_loss
    loss += simclr_loss
    if hasattr(configs.model.esm_encoder,"MLM"):
        if configs.model.esm_encoder.MLM.enable:
            # make masked input and label
            mlm_inputs, mlm_labels = masked_lm_data_collator.mask_tokens(batch_tokens)
            #print(mlm_inputs) have tested
            if hasattr(configs.model.esm_encoder.MLM,"mode") and configs.model.esm_encoder.MLM.mode=="contrast":
                features_seq_mask,_ = simclr.model_seq(mlm_inputs)
                logits_mask, labels_mask = info_nce_loss(features_seq_mask, features_seq,configs.train_settings.n_views,
                                                   configs.train_settings.temperature, accelerator)
                MLM_loss = torch.mean(criterion(logits_mask, labels_mask))
            else:
               prediction_scores = simclr.model_seq(mlm_inputs,return_logits=True)# [seqlen,33]
               # CrossEntropyLoss
               vocab_size = simclr.model_seq.alphabet.all_toks.__len__()
               MLM_loss = torch.mean(criterion(prediction_scores.view(-1, vocab_size), mlm_labels.view(-1)))
            
            loss += MLM_loss
    
    # return loss, simclr_loss,simclr_residue_loss,MLM_loss, logits, labels, logits_residue, labels_residue
    return loss, simclr_loss, MLM_loss, logits, labels

def training_loop_MD(simclr, start_step, train_loader, val_loader, test_loader, batch_converter, criterion,
                  optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
                  result_path, logging, configs, replicate, masked_lm_data_collator=None, **kwargs):
    torch.cuda.empty_cache()
    accelerator = kwargs['accelerator']

    train_loss = 0
    n_steps = start_step
    n_sub_steps = start_step-1
    epoch_num = 0
    """
    train_alt = False
    if hasattr(configs.model.esm_encoder, "MLM"):
        if configs.model.esm_encoder.MLM.enable and configs.model.esm_encoder.MLM.alt:
            train_alt = True
    """
    
    """
    if accelerator.is_main_process:
       seq_evaluation_loop(simclr, n_steps, configs, batch_converter, result_path, valid_writer,logging, accelerator)
       struct_evaluation_loop(simclr, n_steps, configs, result_path, valid_writer,logging, accelerator)
    """
    while True:
        epoch_num += 1
        if accelerator.is_main_process:
            logging.info(f"Epoch {epoch_num}")

        losses = AverageMeter()
        
        progress_bar = tqdm(range(0, int(np.ceil(len(train_loader) / configs.train_settings.gradient_accumulation))),
                            disable=True, leave=True, desc=f"Epoch {epoch_num} steps")
        bsz = configs.train_settings.batch_size
        start = time.time()
        for idx, batch in enumerate(train_loader):
            with accelerator.accumulate(simclr):
                simclr.train()
                batch_seq = [(batch['pid'][i], str(batch['seq'][i])) for i in range(len(batch['seq']))]  # batch['seq']
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
                # graph = batch["graph"]
                trajs = batch["traj"]
                batch_tokens = batch_tokens.to(accelerator.device)
                # graph = graph.to(accelerator.device)
                # trajs = trajs.to(accelerator.device)
                trajs = torch.tensor(np.array(trajs)).to(accelerator.device)
                loss = torch.tensor(0).float()
                loss = loss.to(accelerator.device)
                loss, simclr_loss, MLM_loss,logits, labels= prepare_loss_MD(
                        simclr,trajs,batch_tokens,criterion,loss, accelerator, configs,
                        masked_lm_data_collator
                        )
                    

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(configs.train_settings.batch_size)).mean()
                train_loss += avg_loss.item() / configs.train_settings.gradient_accumulation

                accelerator.backward(loss)

                # Step the optimizers
                if not (configs.model.MD_encoder.fine_tuning.enable is False and configs.model.MD_encoder.fine_tuning_projct.enable is False):
                    optimizer_x.step()
                
                optimizer_seq.step()

                # Step the schedulers
                if not (configs.model.MD_encoder.fine_tuning.enable is False and configs.model.MD_encoder.fine_tuning_projct.enable is False):
                    scheduler_x.step()
                
                scheduler_seq.step()

                losses.update(loss.item(), bsz)

                # Zero the parameter gradients
                if not (configs.model.MD_encoder.fine_tuning.enable is False and configs.model.MD_encoder.fine_tuning_projct.enable is False):
                    optimizer_x.zero_grad()
                optimizer_seq.zero_grad()

            if accelerator.sync_gradients:
                train_writer.add_scalar('step loss', train_loss, n_steps)
                train_writer.add_scalar('learning rate (structure)', optimizer_x.param_groups[0]['lr'], n_steps)
                train_writer.add_scalar('learning rate (sequence)', optimizer_seq.param_groups[0]['lr'], n_steps)
                # writer.add_scalar('loss', loss, global_step=n_steps)
                # writer.add_scalar('acc/top1', top1[0], global_step=n_steps)

                if n_steps % configs.checkpoints_every == 0 and n_steps != 0:
                    # Wait for other processes to catch up
                    print("save_checkpoints")
                    accelerator.wait_for_everyone()
                    # Save checkpoint
                    save_checkpoints(accelerator.unwrap_model(optimizer_x),
                                     accelerator.unwrap_model(optimizer_seq),
                                     result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num)

                # todo: add tensorboard here
                if n_steps % configs.valid_settings.do_every == 0 and n_steps != 0:  # I want to see the step evaluations
                    print("in evaluation loop")
                    # Protein level
                    accelerator.wait_for_everyone()
                    evaluation_loop_MD(simclr, val_loader, labels, batch_converter, criterion, configs,
                                    simclr_loss=simclr_loss, MLM_loss = MLM_loss,losses=losses,
                                    n_steps=n_steps, scheduler_seq=scheduler_seq, result_path=result_path,
                                    scheduler_x=scheduler_x, logits=logits, 
                                    loss=loss, bsz=bsz, train_writer=train_writer,valid_writer=valid_writer,
                                    masked_lm_data_collator=masked_lm_data_collator, logging=logging,
                                    accelerator=accelerator)

                # if configs.valid_settings.eval_struct.enable and (
                #         n_steps % configs.valid_settings.eval_struct.do_every == 0) and n_steps != 0:  # or n_steps == 1):
                #     accelerator.wait_for_everyone()
                #     if accelerator.is_main_process:
                #        struct_evaluation_loop(simclr, n_steps, configs, result_path, valid_writer,logging, accelerator)

                progress_bar.update(1)
                n_steps += 1
                train_loss = 0

            n_sub_steps += 1
            repli_num = configs.train_settings.num_steps if replicate == 2 else (configs.train_settings.num_steps // 3) * (replicate + 1)
            if n_steps > repli_num:
                break

        end = time.time()

        repli_num = configs.train_settings.num_steps if replicate == 2 else (configs.train_settings.num_steps // 3) * (replicate + 1)
        if n_steps > repli_num:
            break

        if accelerator.is_main_process:
            logging.info(f"one epoch cost {(end - start):.2f}, number of trained steps {n_steps}")

    return n_steps, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq

def evaluation_loop_MD(simclr, val_loader, labels, batch_converter, criterion, configs, logging, **kwargs):
    accelerator = kwargs['accelerator']

    n_steps = kwargs['n_steps']
    result_path = kwargs['result_path']
    train_writer = kwargs['train_writer']
    valid_writer = kwargs['valid_writer']

    losses = kwargs['losses']
    loss = kwargs['loss']
    MLM_loss = kwargs['MLM_loss']
    simclr_loss = kwargs['simclr_loss']
    masked_lm_data_collator = kwargs['masked_lm_data_collator']
    logits = kwargs['logits']

    bsz = kwargs['bsz']
    scheduler_struct = kwargs['scheduler_x']
    scheduler_seq = kwargs['scheduler_seq']

    l_prob = logits[:, 0].mean()
    negsim_struct_struct = log_negative_mean_logtis(logits, "struct_struct", bsz)
    negsim_struct_seq = log_negative_mean_logtis(logits, "struct_seq", bsz)
    negsim_seq_seq = log_negative_mean_logtis(logits, "seq_seq", bsz)
    top1, top5 = accuracy(logits, labels, topk=(1, 1))

    if accelerator.is_main_process:
        logging.info(f'evaluation - step {n_steps}')
        logging.info(
            f"step:{n_steps} Loss:{loss:.4f},graph_loss:{simclr_loss:.4f},Top1 accuracy:{top1[0]},P:{l_prob.item():.2f},"
            f"N_st_st:{negsim_struct_struct:.2f},N_st_s:{negsim_struct_seq:.2f},N_s_s:{negsim_seq_seq:.2f}")
        
        if hasattr(configs.model.esm_encoder,"MLM") and configs.model.esm_encoder.MLM.enable:
               logging.info(f"step:{n_steps} MLM_loss: {MLM_loss:.4f}")
               train_writer.add_scalar('mlm_loss', MLM_loss, n_steps)
        
        train_writer.add_scalar('loss', loss, n_steps)
        train_writer.add_scalar('graph_loss', simclr_loss, n_steps)
        train_writer.add_scalar('top_1_accuracy', top1[0], n_steps)
        train_writer.add_scalar('p', l_prob.item(), n_steps)
        train_writer.add_scalar('n_st_st', negsim_struct_struct, n_steps)
        train_writer.add_scalar('n_st_s', negsim_struct_seq, n_steps)
        train_writer.add_scalar('n_s_s', negsim_seq_seq, n_steps)


    loss_val_sum = 0
    graph_loss_sum = 0
    MLM_loss_sum = 0
    l_prob, l_prob_residue = 0, 0
    negsim_struct_seq, negsim_struct_seq_residue = 0, 0
    negsim_seq_seq, negsim_seq_seq_residue = 0, 0
    negsim_struct_struct, negsim_struct_struct_residue = 0, 0

    k = 0
    progress_bar = tqdm(range(0, len(val_loader)), disable=True, leave=True, desc=f"Evaluation steps")
    for batch in val_loader:
        batch_seq = [(batch['pid'][i], str(batch['seq'][i])) for i in range(len(batch['seq']))]  # batch['seq']
        bsz = len(batch)
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
        trajs = batch["traj"]
        batch_tokens = batch_tokens.to(accelerator.device)
        # trajs = trajs.to(accelerator.device)
        trajs = torch.tensor(np.array(trajs)).to(accelerator.device)

        simclr.eval()
        with torch.inference_mode():
            """
            features_struct, residue_struct, features_seq, residue_seq = simclr(graph=graph, batch_tokens=batch_tokens)

            residue_struct, residue_seq, plddt_residue = residue_batch_sample(
                residue_struct, residue_seq, batch['plddt'],
                configs.train_settings.residue_batch_size, accelerator)

            # Prepare loss
            loss_val_sum, graph_loss, residue_loss, logits, labels, logits_residue, labels_residue = prepare_loss(
                features_struct, features_seq, residue_struct, residue_seq, plddt_residue, criterion, (torch.tensor(0).float()).to(accelerator.device),
                accelerator, configs)
            """
            loss_val_sum, graph_loss, MLM_loss,logits, labels= prepare_loss_MD(
                simclr,trajs,batch_tokens,criterion,loss_val_sum,accelerator, configs,
                masked_lm_data_collator
                )
            graph_loss_sum += graph_loss
            MLM_loss_sum += MLM_loss
            l_prob += logits[:, 0].mean().item()
            negsim_struct_struct += log_negative_mean_logtis(logits, "struct_struct", bsz)
            negsim_struct_seq += log_negative_mean_logtis(logits, "struct_seq", bsz)
            negsim_seq_seq += log_negative_mean_logtis(logits, "seq_seq", bsz)

            k = k + 1

        progress_bar.update(1)

    loss_val_sum = loss_val_sum / float(k)
    graph_loss_sum = graph_loss_sum / float(k)
    l_prob = l_prob / float(k)
    negsim_struct_struct = negsim_struct_struct / float(k)
    negsim_struct_seq = negsim_struct_seq / float(k)
    negsim_seq_seq = negsim_seq_seq / float(k)

    (loss_val, val_graph_loss,  val_l_prob, val_negsim_struct_struct, val_negsim_struct_seq,
     val_negsim_seq_seq) = (float(loss_val_sum), float(graph_loss_sum),  float(l_prob),
                                    float(negsim_struct_struct), float(negsim_struct_seq), 
                                    float(negsim_seq_seq))
    
    MLM_loss_sum = float(MLM_loss_sum)/float(k)
    if accelerator.is_main_process:
        logging.info(
            f"step:{n_steps} Val_loss:{loss_val:.4f},val_graph_loss:{val_graph_loss:.4f},val_P:{val_l_prob:.2f},"
            f"val_N_st_st:{val_negsim_struct_struct:.2f},val_N_st_s:{val_negsim_struct_seq:.2f},"
            f"val_N_s_s:{val_negsim_seq_seq:.2f}")
        
        if hasattr(configs.model.esm_encoder,"MLM") and configs.model.esm_encoder.MLM.enable:
               logging.info(f"step:{n_steps} val_MLM_loss: {MLM_loss_sum:.4f}")
               valid_writer.add_scalar('mlm_loss', MLM_loss_sum, n_steps)
        
        valid_writer.add_scalar('loss', loss_val, n_steps)
        valid_writer.add_scalar('graph_loss', val_graph_loss, n_steps)
        valid_writer.add_scalar('p', val_l_prob, n_steps)
        valid_writer.add_scalar('n_st_st', val_negsim_struct_struct, n_steps)
        valid_writer.add_scalar('n_st_s', val_negsim_struct_seq, n_steps)
        valid_writer.add_scalar('n_s_s', val_negsim_seq_seq, n_steps)

        seq_evaluation_loop_MD(simclr, n_steps, configs, batch_converter, result_path, valid_writer, logging, accelerator)

    return n_steps

    # if accelerator.is_main_process:
    #    seq_evaluation_loop(simclr, n_steps, configs, batch_converter, result_path, valid_writer, logging, accelerator)
def seq_evaluation_loop_MD(simclr, n_steps, configs, batch_converter, result_path, valid_writer, logging, accelerator):
    # print("batchsize="+str(int(configs.train_settings.batch_size / 2)))
    logging.info(f'seq evaluation - step {n_steps}')
    simclr.eval()
    scores_cath = evaluate_with_cath_more_md(
        out_figure_path=os.path.join(result_path, 'figures'),
        steps=n_steps,
        batch_size=1,#int(configs.train_settings.batch_size / 2),
        cathpath=configs.valid_settings.cath_path,
        model=simclr,
        accelerator=accelerator,
        batch_converter=batch_converter
    )

    scores_deaminase = evaluate_with_deaminase_md(
        out_figure_path=os.path.join(result_path, 'figures'),
        steps=n_steps,
        batch_size=1, #int(configs.train_settings.batch_size / 2),
        deaminasepath=configs.valid_settings.deaminase_path,
        model=simclr,
        batch_converter=batch_converter,
        accelerator=accelerator
    )

    scores_kinase = evaluate_with_kinase_md(out_figure_path=os.path.join(result_path, 'figures'),
                                         steps=n_steps,
                                         batch_size=1, #int(configs.train_settings.batch_size / 2),
                                         kinasepath=configs.valid_settings.kinase_path,
                                         model=simclr,
                                         batch_converter=batch_converter,
                                         accelerator=accelerator
                                         )
    if accelerator.is_main_process:
        valid_writer.add_scalar('digit_num_1', scores_cath[0], n_steps)
        valid_writer.add_scalar('digit_num_1_2d', scores_cath[1], n_steps)
        
        valid_writer.add_scalar('digit_num_2', scores_cath[4], n_steps)
        valid_writer.add_scalar('digit_num_2_2d', scores_cath[5], n_steps)
        
        valid_writer.add_scalar('digit_num_3', scores_cath[8], n_steps)
        valid_writer.add_scalar('digit_num_3_2d', scores_cath[9], n_steps)
        
        valid_writer.add_scalar('digit_num_1_ARI', scores_cath[2], n_steps)
        valid_writer.add_scalar('digit_num_2_ARI', scores_cath[6], n_steps)
        valid_writer.add_scalar('digit_num_3_ARI', scores_cath[10], n_steps)
        
        valid_writer.add_scalar('digit_num_1_silhouette', scores_cath[3], n_steps)
        valid_writer.add_scalar('digit_num_2_silhouette', scores_cath[7], n_steps)
        valid_writer.add_scalar('digit_num_3_silhouette', scores_cath[11], n_steps)
        
        valid_writer.add_scalar('tdeaminase_score', scores_deaminase[0], n_steps)
        valid_writer.add_scalar('tdeaminase_score_2D', scores_deaminase[1], n_steps)
        valid_writer.add_scalar('tdeaminase_ARI', scores_deaminase[2], n_steps)
        valid_writer.add_scalar('tdeaminase_silhouette', scores_deaminase[3], n_steps)
        
        valid_writer.add_scalar('tkinase_score', scores_kinase[0], n_steps)
        valid_writer.add_scalar('tkinase_score_2D', scores_kinase[1], n_steps)
        valid_writer.add_scalar('tkinase_ARI', scores_kinase[2], n_steps)
        valid_writer.add_scalar('tkinase_silhouette', scores_kinase[3], n_steps)

        logging.info(
            f"step:{n_steps}\tdigit_num_1:{scores_cath[0]:.4f}({scores_cath[1]:.4f})\t"
            f"digit_num_2:{scores_cath[4]:.4f}({scores_cath[5]:.4f})\t"
            f"digit_num_3:{scores_cath[8]:.4f}({scores_cath[9]:.4f})")
        
        logging.info(
            f"step:{n_steps}\tdigit_num_1_ARI:{scores_cath[2]}\tdigit_num_2_ARI:{scores_cath[6]}\t"
            f"digit_num_3_ARI:{scores_cath[10]}")
        
        logging.info(
            f"step:{n_steps}\tdigit_num_1_silhouette:{scores_cath[3]}\tdigit_num_2_silhouette:{scores_cath[7]}\t"
            f"digit_num_3_silhouette:{scores_cath[11]}")
        
        logging.info(
            f"step:{n_steps}\tdeaminase_score:{scores_deaminase[0]:.4f}({scores_deaminase[1]:.4f})\t"
            f"ARI:{scores_deaminase[2]:.4f} silhouette:{scores_deaminase[3]:.4f}")
        logging.info(
            f"step:{n_steps}\tkinase_score:{scores_kinase[0]:.4f}({scores_kinase[1]:.4f})\t"
            f"ARI:{scores_kinase[2]:.4f} silhouette:{scores_kinase[3]:.4f}")

def evaluation_loop(simclr, val_loader, labels, labels_residue, batch_converter, criterion, configs, logging, **kwargs):
    accelerator = kwargs['accelerator']

    n_steps = kwargs['n_steps']
    result_path = kwargs['result_path']
    train_writer = kwargs['train_writer']
    valid_writer = kwargs['valid_writer']

    losses = kwargs['losses']
    loss = kwargs['loss']
    MLM_loss = kwargs['MLM_loss']
    simclr_loss = kwargs['simclr_loss']
    simclr_residue_loss = kwargs['simclr_residue_loss']
    masked_lm_data_collator = kwargs['masked_lm_data_collator']
    logits = kwargs['logits']
    logits_residue = kwargs['logits_residue']

    bsz = kwargs['bsz']
    scheduler_struct = kwargs['scheduler_struct']
    scheduler_seq = kwargs['scheduler_seq']

    l_prob = logits[:, 0].mean()
    negsim_struct_struct = log_negative_mean_logtis(logits, "struct_struct", bsz)
    negsim_struct_seq = log_negative_mean_logtis(logits, "struct_seq", bsz)
    negsim_seq_seq = log_negative_mean_logtis(logits, "seq_seq", bsz)
    top1, top5 = accuracy(logits, labels, topk=(1, 1))

    # residue_level
    l_prob_residue = logits_residue[:, 0].mean()
    bsz_residue = int(len(logits_residue) / 2)
    negsim_struct_struct_residue = log_negative_mean_logtis(logits_residue, "struct_struct", bsz_residue)
    negsim_struct_seq_residue = log_negative_mean_logtis(logits_residue, "struct_seq", bsz_residue)
    negsim_seq_seq_residue = log_negative_mean_logtis(logits_residue, "seq_seq", bsz_residue)
    top1_residue, top5 = accuracy(logits_residue, labels_residue, topk=(1, 1))

    if accelerator.is_main_process:
        logging.info(f'evaluation - step {n_steps}')
        logging.info(
            f"step:{n_steps} Loss:{loss:.4f},graph_loss:{simclr_loss:.4f},Top1 accuracy:{top1[0]},P:{l_prob.item():.2f},"
            f"N_st_st:{negsim_struct_struct:.2f},N_st_s:{negsim_struct_seq:.2f},N_s_s:{negsim_seq_seq:.2f}")
        
        if hasattr(configs.model.esm_encoder,"MLM") and configs.model.esm_encoder.MLM.enable:
               logging.info(f"step:{n_steps} MLM_loss: {MLM_loss:.4f}")
               train_writer.add_scalar('mlm_loss', MLM_loss, n_steps)
        
        train_writer.add_scalar('loss', loss, n_steps)
        train_writer.add_scalar('graph_loss', simclr_loss, n_steps)
        train_writer.add_scalar('top_1_accuracy', top1[0], n_steps)
        train_writer.add_scalar('p', l_prob.item(), n_steps)
        train_writer.add_scalar('n_st_st', negsim_struct_struct, n_steps)
        train_writer.add_scalar('n_st_s', negsim_struct_seq, n_steps)
        train_writer.add_scalar('n_s_s', negsim_seq_seq, n_steps)

        logging.info(
            f"step:{n_steps} Loss:{loss:.4f},residue_loss:{simclr_residue_loss:.4f},Top1 accuracy:{top1_residue[0]},"
            f"P:{l_prob_residue.item():.2f},N_st_st:{negsim_struct_struct_residue:.2f},"
            f"N_st_s:{negsim_struct_seq_residue:.2f},N_s_s:{negsim_seq_seq_residue:.2f}")

        train_writer.add_scalar('residue_loss', simclr_residue_loss, n_steps)
        train_writer.add_scalar('top_1_accuracy_residue', top1_residue[0], n_steps)
        train_writer.add_scalar('p_residue', l_prob_residue.item(), n_steps)
        train_writer.add_scalar('n_st_st_residue', negsim_struct_struct_residue, n_steps)
        train_writer.add_scalar('n_st_s_residue', negsim_struct_seq_residue, n_steps)
        train_writer.add_scalar('n_s_s_residue', negsim_seq_seq_residue, n_steps)

    loss_val_sum = 0
    residue_loss_sum = 0
    graph_loss_sum = 0
    MLM_loss_sum = 0
    l_prob, l_prob_residue = 0, 0
    negsim_struct_seq, negsim_struct_seq_residue = 0, 0
    negsim_seq_seq, negsim_seq_seq_residue = 0, 0
    negsim_struct_struct, negsim_struct_struct_residue = 0, 0

    k = 0
    progress_bar = tqdm(range(0, len(val_loader)), disable=True, leave=True, desc=f"Evaluation steps")
    for batch in val_loader:
        batch_seq = [(batch['pid'][i], str(batch['seq'][i])) for i in range(len(batch['seq']))]  # batch['seq']
        bsz = len(batch)
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
        if configs.model.X_module=="Geom2vec_tematt" or configs.model.X_module=="Geom2vec_temlstmatt":
            res_rep = batch['res_rep']
            batch_idx = batch['batch_idx']
            graph = [res_rep.to(accelerator.device), batch_idx.to(accelerator.device)]
        else:
            graph = batch["graph"]
            graph = graph.to(accelerator.device)
        batch_tokens = batch_tokens.to(accelerator.device)
        

        simclr.eval()
        with torch.inference_mode():
            """
            features_struct, residue_struct, features_seq, residue_seq = simclr(graph=graph, batch_tokens=batch_tokens)

            residue_struct, residue_seq, plddt_residue = residue_batch_sample(
                residue_struct, residue_seq, batch['plddt'],
                configs.train_settings.residue_batch_size, accelerator)

            # Prepare loss
            loss_val_sum, graph_loss, residue_loss, logits, labels, logits_residue, labels_residue = prepare_loss(
                features_struct, features_seq, residue_struct, residue_seq, plddt_residue, criterion, (torch.tensor(0).float()).to(accelerator.device),
                accelerator, configs)
            """
            loss_val_sum, graph_loss, residue_loss, MLM_loss,logits, labels, logits_residue, labels_residue = prepare_loss(
                simclr,graph,batch_tokens,None,criterion,loss_val_sum,accelerator, configs,
                masked_lm_data_collator
                )
            graph_loss_sum += graph_loss
            residue_loss_sum += residue_loss
            MLM_loss_sum += MLM_loss
            l_prob += logits[:, 0].mean().item()
            negsim_struct_struct += log_negative_mean_logtis(logits, "struct_struct", bsz)
            negsim_struct_seq += log_negative_mean_logtis(logits, "struct_seq", bsz)
            negsim_seq_seq += log_negative_mean_logtis(logits, "seq_seq", bsz)

            bsz_residue = int(len(logits_residue) / 2)
            l_prob_residue += logits_residue[:, 0].mean().item()
            negsim_struct_struct_residue += log_negative_mean_logtis(logits_residue, "struct_struct", bsz_residue)
            negsim_struct_seq_residue += log_negative_mean_logtis(logits_residue, "struct_seq", bsz_residue)
            negsim_seq_seq_residue += log_negative_mean_logtis(logits_residue, "seq_seq", bsz_residue)
            k = k + 1

        progress_bar.update(1)

    loss_val_sum = loss_val_sum / float(k)
    graph_loss_sum = graph_loss_sum / float(k)
    residue_loss_sum = residue_loss_sum / float(k)
    l_prob = l_prob / float(k)
    negsim_struct_struct = negsim_struct_struct / float(k)
    negsim_struct_seq = negsim_struct_seq / float(k)
    negsim_seq_seq = negsim_seq_seq / float(k)

    l_prob_residue = l_prob_residue / float(k)
    negsim_struct_struct_residue = negsim_struct_struct_residue / float(k)
    negsim_struct_seq_residue = negsim_struct_seq_residue / float(k)
    negsim_seq_seq_residue = negsim_seq_seq_residue / float(k)

    (loss_val, val_graph_loss, val_residue_loss, val_l_prob, val_negsim_struct_struct, val_negsim_struct_seq,
     val_negsim_seq_seq, val_l_prob_residue, val_negsim_struct_struct_residue, val_negsim_struct_seq_residue,
     val_negsim_seq_seq_residue) = (float(loss_val_sum), float(graph_loss_sum), float(residue_loss_sum), float(l_prob),
                                    float(negsim_struct_struct), float(negsim_struct_seq), float(negsim_seq_seq),
                                    float(l_prob_residue), float(negsim_struct_struct_residue),
                                    float(negsim_struct_seq_residue), float(negsim_seq_seq_residue))
    
    MLM_loss_sum = float(MLM_loss_sum)/float(k)
    if accelerator.is_main_process:
        logging.info(
            f"step:{n_steps} Val_loss:{loss_val:.4f},val_graph_loss:{val_graph_loss:.4f},val_P:{val_l_prob:.2f},"
            f"val_N_st_st:{val_negsim_struct_struct:.2f},val_N_st_s:{val_negsim_struct_seq:.2f},"
            f"val_N_s_s:{val_negsim_seq_seq:.2f}")
        
        if hasattr(configs.model.esm_encoder,"MLM") and configs.model.esm_encoder.MLM.enable:
               logging.info(f"step:{n_steps} val_MLM_loss: {MLM_loss_sum:.4f}")
               valid_writer.add_scalar('mlm_loss', MLM_loss_sum, n_steps)
        
        valid_writer.add_scalar('loss', loss_val, n_steps)
        valid_writer.add_scalar('graph_loss', val_graph_loss, n_steps)
        valid_writer.add_scalar('p', val_l_prob, n_steps)
        valid_writer.add_scalar('n_st_st', val_negsim_struct_struct, n_steps)
        valid_writer.add_scalar('n_st_s', val_negsim_struct_seq, n_steps)
        valid_writer.add_scalar('n_s_s', val_negsim_seq_seq, n_steps)

        logging.info(
            f"step:{n_steps} Val_loss:{loss_val:.4f},val_residue_loss:{val_residue_loss:.4f},"
            f"val_P:{val_l_prob_residue:.2f},val_N_st_st:{val_negsim_struct_struct_residue:.2f},"
            f"val_N_st_s:{val_negsim_struct_seq_residue:.2f},val_N_s_s:{val_negsim_seq_seq_residue:.2f}")
        logging.info(
            f"step:{n_steps}\tTrain_loss_avg: {losses.avg:.6f}, lr: {scheduler_struct.get_lr()[0]:.2e}")

        valid_writer.add_scalar('residue_loss', val_residue_loss, n_steps)
        valid_writer.add_scalar('p_residue', val_l_prob_residue, n_steps)
        valid_writer.add_scalar('n_st_st_residue', val_negsim_struct_struct_residue, n_steps)
        valid_writer.add_scalar('n_st_s_residue', val_negsim_struct_seq_residue, n_steps)
        valid_writer.add_scalar('n_s_s_residue', val_negsim_seq_seq_residue, n_steps)
    if accelerator.is_main_process:
       seq_evaluation_loop(simclr, n_steps, configs, batch_converter, result_path, valid_writer, logging, accelerator)
    return loss_val, val_graph_loss, val_residue_loss


def seq_evaluation_loop(simclr, n_steps, configs, batch_converter, result_path, valid_writer, logging, accelerator):
    # print("batchsize="+str(int(configs.train_settings.batch_size / 2)))
    simclr.eval()
    scores_cath = evaluate_with_cath_more(
        out_figure_path=os.path.join(result_path, 'figures'),
        steps=n_steps,
        batch_size=1, #int(configs.train_settings.batch_size / 2),
        cathpath=configs.valid_settings.cath_path,
        model=simclr,
        accelerator=accelerator,
        batch_converter=batch_converter
    )

    scores_deaminase = evaluate_with_deaminase(
        out_figure_path=os.path.join(result_path, 'figures'),
        steps=n_steps,
        batch_size=1, #int(configs.train_settings.batch_size / 2),
        deaminasepath=configs.valid_settings.deaminase_path,
        model=simclr,
        batch_converter=batch_converter,
        accelerator=accelerator
    )

    scores_kinase = evaluate_with_kinase(out_figure_path=os.path.join(result_path, 'figures'),
                                         steps=n_steps,
                                         batch_size=1, #int(configs.train_settings.batch_size / 2),
                                         kinasepath=configs.valid_settings.kinase_path,
                                         model=simclr,
                                         batch_converter=batch_converter,
                                         accelerator=accelerator
                                         )
    if accelerator.is_main_process:
        logging.info(
            f"step:{n_steps}\tdigit_num_1:{scores_cath[0]:.4f}({scores_cath[1]:.4f})\t"
            f"digit_num_2:{scores_cath[3]:.4f}({scores_cath[4]:.4f})\t"
            f"digit_num_3:{scores_cath[6]:.4f}({scores_cath[7]:.4f})")
        
        valid_writer.add_scalar('digit_num_1', scores_cath[0], n_steps)
        valid_writer.add_scalar('digit_num_1_2d', scores_cath[1], n_steps)
        valid_writer.add_scalar('digit_num_2', scores_cath[3], n_steps)
        valid_writer.add_scalar('digit_num_2_2d', scores_cath[4], n_steps)
        valid_writer.add_scalar('digit_num_3', scores_cath[6], n_steps)
        valid_writer.add_scalar('digit_num_3_2d', scores_cath[7], n_steps)
        
        logging.info(
            f"step:{n_steps}\tdigit_num_1_ARI:{scores_cath[2]}\tdigit_num_2_ARI:{scores_cath[5]}\t"
            f"digit_num_3_ARI:{scores_cath[8]}")
        valid_writer.add_scalar('digit_num_1_ARI', scores_cath[2], n_steps)
        valid_writer.add_scalar('digit_num_2_ARI', scores_cath[5], n_steps)
        valid_writer.add_scalar('digit_num_3_ARI', scores_cath[8], n_steps)
        logging.info(
            f"step:{n_steps}\tdeaminase_score:{scores_deaminase[0]:.4f}({scores_deaminase[1]:.4f})\t"
            f"ARI:{scores_deaminase[2]}")
        logging.info(
            f"step:{n_steps}\tkinase_score:{scores_kinase[0]:.4f}({scores_kinase[1]:.4f})\tARI:{scores_kinase[2]}")
        
        valid_writer.add_scalar('tdeaminase_score', scores_deaminase[0], n_steps)
        valid_writer.add_scalar('tdeaminase_score_2D', scores_deaminase[1], n_steps)
        valid_writer.add_scalar('tdeaminase_ARI', scores_deaminase[2], n_steps)
        valid_writer.add_scalar('tkinase_score', scores_kinase[0], n_steps)
        valid_writer.add_scalar('tkinase_score_2D', scores_kinase[1], n_steps)
        valid_writer.add_scalar('tkinase_ARI', scores_kinase[2], n_steps)

        

def struct_evaluation_loop(simclr, n_steps, configs, result_path, valid_writer, logging, accelerator):
    simclr.eval()
    logging.info(f'structure evaluation - step {n_steps}')
    scores_cath = evaluate_with_cath_more_struct(
        accelerator=accelerator,
        batch_size=int(configs.train_settings.batch_size / 2),
        out_figure_path=os.path.join(result_path, 'figures'),
        steps=n_steps,
        cathpath=configs.valid_settings.eval_struct.cath_pdb_path,
        model=simclr,
        configs = configs
        #seq_mode=configs.model.struct_encoder.use_seq.seq_embed_mode
    )

    logging.info(
        f"step:{n_steps}\tgvp_digit_num_1:{scores_cath[0]:.4f}({scores_cath[1]:.4f})"
        f"\tgvp_digit_num_2:{scores_cath[3]:.4f}({scores_cath[4]:.4f})"
        f"\tgvp_digit_num_3:{scores_cath[6]:.4f}({scores_cath[7]:.4f})")
    
    valid_writer.add_scalar('gvp_digit_num_1', scores_cath[0], n_steps)
    valid_writer.add_scalar('gvp_digit_num_1_2d', scores_cath[1], n_steps)
    valid_writer.add_scalar('gvp_digit_num_2', scores_cath[3], n_steps)
    valid_writer.add_scalar('gvp_digit_num_2_2d', scores_cath[4], n_steps)
    valid_writer.add_scalar('gvp_digit_num_3', scores_cath[6], n_steps)
    valid_writer.add_scalar('gvp_digit_num_3_2d', scores_cath[7], n_steps)
        
    logging.info(
        f"step:{n_steps}\tgvp_digit_num_1_ARI:{scores_cath[2]}\tgvp_digit_num_2_ARI:{scores_cath[5]}"
        f"\tgvp_digit_num_3_ARI:{scores_cath[8]}")
    
    valid_writer.add_scalar('gvp_digit_num_1_ARI', scores_cath[2], n_steps)
    valid_writer.add_scalar('gvp_digit_num_2_ARI', scores_cath[5], n_steps)
    valid_writer.add_scalar('gvp_digit_num_3_ARI', scores_cath[8], n_steps)

def main(args, dict_configs, config_file_path):
    configs = load_configs(dict_configs, args)
    if args.seed:
        configs.fix_seed = args.seed
    
    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()

    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)

    logging = get_logging(result_path)

    train_writer, valid_writer = prepare_tensorboard(result_path)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.gradient_accumulation,
        dispatch_batches=False,
        kwargs_handlers=[ddp_kwargs]
    )
    if accelerator.is_main_process:
        test_gpu_cuda()

    simclr = prepare_models(logging, configs, accelerator)
    if accelerator.is_main_process:
        logging.info('preparing model is done')
    
    scheduler_seq, scheduler_x, optimizer_seq, optimizer_x = prepare_optimizer(
        simclr.model_seq, simclr.model_x, logging, configs
    )

    if accelerator.is_main_process:
        logging.info('preparing optimizer is done')
    
    start_step = 0
    if configs.resume.resume:
        if configs.model.X_module == 'MD' or configs.model.X_module=='DCCM_GNN':
            simclr,start_step, best_score = load_checkpoints_md(simclr, configs, 
                            optimizer_seq,optimizer_x,scheduler_seq,scheduler_x,
                            logging,resume_path=configs.resume.resume_path,restart_optimizer=configs.resume.restart_optimizer)
        if configs.model.X_module == 'structure':
            simclr,start_step = load_checkpoints(simclr, configs, 
                            optimizer_seq,optimizer_x,scheduler_seq,scheduler_x,
                            logging,resume_path=configs.resume.result_path,restart_optimizer=configs.resume.restart_optimizer)
        if configs.model.X_module == 'DCCM_GNN':
            print("to do...")
            #xxx
        if configs.model.X_module == 'Geom2vec':
            print("to do...")
        if configs.model.X_module == 'Geom2vec_tematt':
            print("to do...")
        if configs.model.X_module == 'Geom2vec_temlstmatt':
            print("to do...")
    
    alphabet = simclr.model_seq.alphabet
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=configs.model.esm_encoder.max_length)

    if hasattr(configs.model.esm_encoder, "MLM") and configs.model.esm_encoder.MLM.enable:
        masked_lm_data_collator = MaskedLMDataCollator(batch_converter,
                                                       mlm_probability=configs.model.esm_encoder.MLM.mask_ratio)
    else:
        masked_lm_data_collator = None

    if configs.model.X_module == 'MD':
        from data.data_MD import prepare_dataloaders
        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
         (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
         (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)) = prepare_dataloaders(configs)
        # val_loader = prepare_dataloaders_v2(configs)
        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
        (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
        (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2))=accelerator.prepare(((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
                                                                                                        (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
                                                                                                        (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)))
    if configs.model.X_module == 'vivit':
        from data.data_vivit import prepare_dataloaders
        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
         (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
         (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)) = prepare_dataloaders(configs)
        # val_loader = prepare_dataloaders_v2(configs)
        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
        (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
        (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2))=accelerator.prepare(((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
                                                                                                        (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
                                                                                                        (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)))
    elif configs.model.X_module == 'DCCM_GNN':
        from data.data_MD_feature_GNN import prepare_dataloaders
        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
         (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
         (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)) = prepare_dataloaders(configs)
        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
        (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
        (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2))=accelerator.prepare(((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
                                                                                                        (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
                                                                                                        (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)))
    elif configs.model.X_module == 'Geom2vec':
        from data.data_geom2vec import prepare_dataloaders
        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
         (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
         (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)) = prepare_dataloaders(configs)
        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
        (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
        (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2))=accelerator.prepare(((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
                                                                                                        (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
                                                                                                        (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)))
    elif configs.model.X_module == 'Geom2vec_tematt' or configs.model.X_module == 'Geom2vec_temlstmatt':
        from data.data_geom2vec_tematt import prepare_dataloaders
        # train_loader, val_loader, test_loader = prepare_dataloaders(configs)
        # train_loader, val_loader, test_loader=accelerator.prepare(train_loader, val_loader, test_loader)
        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
         (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
         (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)) = prepare_dataloaders(configs)
        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
        (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
        (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2))=accelerator.prepare(((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
                                                                                                        (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
                                                                                                        (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)))
        
    elif configs.model.X_module == 'structure':
        if hasattr(configs.model.struct_encoder,"version") and configs.model.struct_encoder.version == 'v2':
            from data.data_v2 import prepare_dataloaders as prepare_dataloaders_v2
            train_loader, val_loader = prepare_dataloaders_v2(logging, accelerator, configs)
        else:
            from data.data import prepare_dataloaders
            train_loader, val_loader = prepare_dataloaders(logging, accelerator, configs)
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    if accelerator.is_main_process:
        logging.info('preparing dataloaders are done')

    # Register modules with accelerate
    simclr, scheduler_seq, scheduler_x, optimizer_seq, optimizer_x = accelerator.prepare(
        simclr, scheduler_seq, scheduler_x, optimizer_seq, optimizer_x
        )
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if accelerator.is_main_process:
        logging.info(f"Start contrastive training for {configs.train_settings.num_steps} steps.")

    if accelerator.is_main_process:
        if configs.model.X_module == 'structure':
            train_steps = np.ceil(len(train_loader) / configs.train_settings.gradient_accumulation)
        elif configs.model.X_module == 'MD' or \
            configs.model.X_module == 'DCCM_GNN' or \
                configs.model.X_module == 'Geom2vec_tematt' or \
                    configs.model.X_module == 'Geom2vec_temlstmatt' or \
                        configs.model.X_module == 'vivit':
            train_steps = np.ceil(len(train_dataloader_repli_0) / configs.train_settings.gradient_accumulation)
            train_steps = train_steps * 3
        logging.info(f'Number of train steps per epoch: {int(train_steps)}')
        logging.info(f"Training with: {accelerator.device} and fix_seed = {configs.fix_seed}")
    
    if configs.model.X_module == 'structure':
        training_loop(
            simclr, start_step,train_loader, val_loader, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator
        )
    elif configs.model.X_module == 'Geom2vec_tematt' or configs.model.X_module == 'Geom2vec_temlstmatt':
        # training_loop_Geom2ve_tematt(
        #     simclr, start_step,train_loader, val_loader, test_loader, batch_converter, criterion,
        #     optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
        #     result_path, logging, configs, masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator
        # )
        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq = training_loop_Geom2ve_tematt(
            simclr, start_step, train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, replicate=0, masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator
        )
        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq = training_loop_Geom2ve_tematt(
            simclr, start_step, train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, replicate=1, masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator
        )
        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq = training_loop_Geom2ve_tematt(
            simclr, start_step, train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, replicate=2, masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator
        )

    elif configs.model.X_module == 'MD' or configs.model.X_module == 'vivit':
        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq = training_loop_MD(
            simclr, start_step, train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, replicate=0, masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator)
        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq = training_loop_MD(
            simclr, start_step, train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, replicate=1, masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator)
        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq = training_loop_MD(
            simclr, start_step, train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, replicate=2, masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator)
    elif configs.model.X_module == 'DCCM_GNN':
        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq = training_loop_DCCM_GNN(
            simclr, start_step, train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, replicate=0, masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator)
        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq = training_loop_DCCM_GNN(
            simclr, start_step, train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, replicate=1, masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator)
        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq = training_loop_DCCM_GNN(
            simclr, start_step, train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, train_writer, valid_writer,
            result_path, logging, configs, replicate=2, masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator)

    train_writer.close()
    valid_writer.close()

    accelerator.free_memory()
    torch.cuda.empty_cache()
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./config.yaml')
    parser.add_argument("--result_path", default=None,
                        help="result_path, if set by command line, overwrite the one in config.yaml, "
                             "by default is None")
    parser.add_argument("--resume_path", default=None,
                        help="if set, overwrite the one in config.yaml, by default is None")
    parser.add_argument("--num_end_adapter_layers", default=None, help="num_end_adapter_layers")
    parser.add_argument("--module_type", default=None, help="module_type for adapterh")
    parser.add_argument("--seed",default=None,type=int,help="random seed")
    parser.add_argument("--restart_optimizer",default=None,type=int,help="restart_optimizer")
    
    args_main = parser.parse_args()
    config_path = args_main.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(args_main, config_file, config_path)
