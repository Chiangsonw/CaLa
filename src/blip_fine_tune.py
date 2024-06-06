from comet_ml import Experiment
import json
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
import clip
from lavis.models import load_model
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, \
    extract_index_features_blip2, generate_randomized_fiq_caption, extract_index_features_clip, \
    device, element_wise_sum, cosine_lr_schedule 
from validate import compute_cirr_val_metrics_blip2, compute_fiq_val_metrics_blip2, compute_fiq_val_metrics_clip, \
artemis_compute_cirr_val_metrics ,compute_cirr_val_metrics_clip, artemis_compute_fiq_val_metrics
from twin_attention_compositor_blip2 import TwinAttentionCompositorBLIP2
from hinge_based_cross_attention_blip2 import HingebasedCrossAttentionBLIP2
from twin_attention_compositor_clip import TwinAttentionCompositorCLIP
from hinge_based_cross_attention_clip import HingebasedCrossAttentionCLIP
import random
from artemis import Artemis
import ssl

base_path = Path(__file__).absolute().parents[1].absolute()

def blip_finetune_fiq(train_dress_types: List[str], val_dress_types: List[str],
                      num_epochs: int, batch_size: int,
                      validation_frequency: int, transform: str, save_training: bool, save_best: bool,
                      **kwargs):
    """
    Fine-tune blip on the FashionIQ dataset using as combining function the image-text element-wise sum
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param num_epochs: number of epochs
    :param blip_model_name: blip model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning leanring rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['blip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the fine-tuned blip model
    :param encoder: which blip encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best blip model wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg
    """

    experiment_name = kwargs["experiment_name"]
    # training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/blip_cirr_{experiment_name}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    
    # initialize encoders
    encoder_arch = kwargs["encoder_arch"]
    model_name = kwargs["model_name"]

    if encoder_arch == "blip2":
        blip_textual = load_model(name=model_name, model_type="pretrain_vitL", is_eval=True, device=device)
        blip_multimodal = load_model(name=model_name, model_type="pretrain_vitL", is_eval=True, device=device)
        blip_visual = load_model(name=model_name, model_type="pretrain_vitL", is_eval=True, device=device)
    
    elif encoder_arch == "clip":
        clip_model, clip_preprocess = clip.load(model_name, device=device, jit=False)
        clip_model.eval().float()

    # initialize support modules
    embeds_dim = kwargs["embeds_dim"]
    if encoder_arch == "blip2":
        tac = TwinAttentionCompositorBLIP2(embeds_dim).to(device)
        hca = HingebasedCrossAttentionBLIP2(embeds_dim).to(device)
    
    elif encoder_arch == "clip":
        tac = TwinAttentionCompositorCLIP().to(device)
        hca = HingebasedCrossAttentionCLIP(embeds_dim).to(device)
    
    cir_frame = kwargs["cir_frame"]
    artemis = Artemis(embeds_dim).to(device)


    # define the combining func
    combining_function = element_wise_sum
    # preprocess
    if encoder_arch == "blip2":
        input_dim = 224
    elif encoder_arch == "clip":
        input_dim = clip_model.visual.input_resolution


    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['blip', 'squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs
    encoder = kwargs["encoder"]
    if encoder == 'text':
        index_features_list = []
        index_names_list = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)
        if encoder == 'text':
            if encoder_arch == "blip2":
                index_features_and_names = extract_index_features_blip2(classic_val_dataset, clip_model)
                index_features_list.append(index_features_and_names[0])
                index_names_list.append(index_features_and_names[1])

            if encoder_arch == "clip":
                index_features_and_names = extract_index_features_clip(classic_val_dataset, clip_model)
                index_features_list.append(index_features_and_names[0])
                index_names_list.append(index_features_and_names[1])


    # Define the train datasets and the combining function
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=0, pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    learning_rate = kwargs["learning_rate"]
    min_lr = kwargs["min_lr"]
    max_epoch = kwargs["max_epoch"]
                                                         

    # Define the optimizer, the loss and the grad scaler
    if encoder_arch == "blip2": 
        # # blip2_encoder
        if encoder == 'multi': # only finetuning text_encoder
            optimizer = optim.AdamW([  # param in blip_multimodal
                {'params': [param for param in blip_multimodal.Qformer.bert.parameters()], 
                'lr': learning_rate,
                'weight_decay': 0.05},
                {'params': blip_multimodal.query_tokens,
                'lr': learning_rate,
                'weight_decay': 0.05},
                # param in blip_textual
                {'params': [param for param in blip_textual.Qformer.bert.parameters()],
                'lr': learning_rate, 
                'weight_decay': 0.05},

                 # params in support modules
                {'params': [param for param in tac.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},

                {'params': [param for param in hca.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},

                {'params': [param for param in artemis.parameters()], 
                'lr': learning_rate * 10,
                'weight_decay': 0.05},
                ])
                   
        elif encoder == 'both': #  finetuning textual and visual concurrently
            optimizer = optim.AdamW([
                # param in blip_multimodal
                {'params': [param for param in blip_multimodal.Qformer.bert.parameters()], 
                'lr': learning_rate,
                'weight_decay': 0.05},
                {'params': blip_multimodal.query_tokens,
                'lr': learning_rate,
                'weight_decay': 0.05},

                # param in blip_textual
                {'params': [param for param in blip_textual.Qformer.bert.parameters()],
                'lr': learning_rate,
                'weight_decay': 0.05},

                # param in blip_visual
                {'params': [param for param in blip_visual.Qformer.bert.parameters()], 
                'lr': learning_rate,
                'weight_decay': 0.05},
                {'params': blip_visual.query_tokens,
                'lr': learning_rate,
                'weight_decay': 0.05},

                # params in support modules

                {'params': [param for param in tac.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},

                {'params': [param for param in hca.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},

                {'params': [param for param in artemis.parameters()], 
                'lr': learning_rate * 10,
                'weight_decay': 0.05},
                ])

        else:
            raise ValueError("encoders to finetune must be 'multi' or 'both'")
    elif encoder_arch == "clip": 
        # clip_encoder
        optimizer = optim.AdamW([  # param in blip_multimodal
                {'params': [param for name, param in clip_model.named_parameters()
                        if 'visual' not in name], 
                'lr': learning_rate,
                'weight_decay': 0.05},

                 # params in support modules
                {'params': [param for param in tac.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},

                {'params': [param for param in hca.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},
                ])

    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # define weights for different modules
    tac_weight = kwargs["tac_weight"]
    hca_weight = kwargs["hca_weight"]

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        with experiment.train():
            # encoder = "text" or "both"
            # set models to train mode
            if encoder_arch == "blip2":
                # # blip2_encoder
                blip_multimodal.Qformer.bert.train()
                # blip_multimodal.vision_proj.train()
                blip_multimodal.query_tokens.requires_grad = True
                blip_textual.Qformer.bert.train()
                # blip_textual.text_proj.train()

                # both adds param in visual_encoder
                # blip2_encoder
                if encoder == "both":
                    blip_visual.Qformer.bert.train()
                    # blip_visual.vision_proj.train()
                    blip_visual.query_tokens.requires_grad = True
            
            elif encoder_arch == "clip":
                # clip_encoder            
                clip_model.train()
       
            # support modules
            if tac_weight > 0:
                tac.train()
            if hca_weight > 0:
                hca.train()
            if cir_frame == "artemis":  
                artemis.train()


            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            train_bar = tqdm(relative_train_loader, ncols=150)
            
            # adjust learning rate 
            cosine_lr_schedule(optimizer, epoch, max_epoch, learning_rate, min_lr, onlyGroup0=True)
            
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):
                images_in_batch = reference_images.size(0)
                step = len(train_bar) * epoch + idx
                optimizer.zero_grad()

                # move ref and tar img to device
                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)

                # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                flattened_captions: list = np.array(captions).T.flatten().tolist()
                input_captions = generate_randomized_fiq_caption(flattened_captions)

                # Extract the features, compute the logits and the loss

                with torch.cuda.amp.autocast():
                    if encoder_arch == "blip2":
                        # text
                        text_embeds = blip_textual.extract_features({"text_input":input_captions},
                                                                mode="text").text_embeds
                        text_feats = text_embeds[:,0,:]
                        
                        # target
                        target_embeds = blip_visual.extract_features({"image":target_images}, 
                                                                    mode="image").image_embeds
                        target_feats = F.normalize(target_embeds[:,0,:], dim=-1)

                        # reference
                        reference_embeds = blip_multimodal.extract_features({"image":reference_images,
                                                                    "text_input":input_captions}).multimodal_embeds
                        reference_feats = reference_embeds[:,0,:]
                        
                        # embeds encoded with visual_encoder
                        reference_embeds_for_tac = blip_visual.extract_features({"image":reference_images}, 
                                                                    mode="image").image_embeds
                    elif encoder_arch == "clip":
                        # reference
                        reference_feats, reference_embeds = clip_model.encode_image(reference_images)
                        reference_embeds_for_tac = reference_embeds
                        
                        # text
                        text_inputs = clip.tokenize(input_captions, context_length=77, truncate=True).to(device, non_blocking=True)
                        text_feats, text_embeds = clip_model.encode_text(text_inputs)
                        
                        # target
                        target_feats, target_embeds = clip_model.encode_image(target_images)
                        target_feats = F.normalize(target_feats)

                    # ============ Query-Target Contrastive =========== 
                    
                    if cir_frame == "artemis":
                        # artemis
                        artemis_scores =  artemis.compute_score_broadcast_artemis(reference_feats, text_feats, target_feats)
                        # artemis_logits = artemis.temperature.exp() * artemis_scores
                    elif cir_frame == "sum":
                        # sum_predicted
                        predicted_feats = combining_function(reference_feats, text_feats)
                        matching_logits = 100 * predicted_feats @ target_feats.T
                    # ============ Query-Target Align =========== 
                        
                    # align(tac)
                    if tac_weight > 0:
                        visual_gap_feats = tac(reference_embeds_for_tac, target_embeds)
                        aligning_logits = 10 * text_feats @ visual_gap_feats.T

                    # ============ Reference-Caption-Target Contrastive =========== 
                    if hca_weight > 0:
                        psudo_T = hca(reference_embeds = reference_embeds,
                                            caption_embeds = text_embeds,
                                            target_embeds = target_embeds)
                        
                        reasoning_logits = 10 * psudo_T @ reference_feats.T 

                    
                   # ============ LOSS =========== 
                    # align_loss / tac
                    # align_loss = crossentropy_criterion(align_logits, ground_truth)

                    # hca_loss
                    # hca_tcr_loss = crossentropy_criterion(hca_logits, ground_truth)
                    
                    if cir_frame == "artemis":
                        # artemis_loss
                        # artemis_loss = crossentropy_criterion(artemis_logits, ground_truth)
                        if tac_weight > 0:
                            contrastive_logits = tac_weight * aligning_logits + \
                                            (1 - tac_weight) * artemis_scores
                        else:
                            contrastive_logits = artemis_scores
                    elif cir_frame == "sum":
                        # contrastive loss
                        # contrastive_loss = crossentropy_criterion(contrast_logits, ground_truth)
                        if tac_weight > 0:
                            contrastive_logits = tac_weight * aligning_logits + \
                                            (1 - tac_weight) * matching_logits
                        else:
                            contrastive_logits = matching_logits

                    # ========== Sum_Loss  ===============
                    # hca_loss
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    contrastive_loss = crossentropy_criterion(contrastive_logits, ground_truth)
                    if hca_weight > 0:
                        reasoning_loss = crossentropy_criterion(reasoning_logits, ground_truth)
                        loss = hca_weight * reasoning_loss + (1 - hca_weight) * contrastive_loss
                    loss = contrastive_loss

                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            with experiment.validate():
                recalls_at10 = []
                recalls_at50 = []

                # Compute and log validation metrics for each validation dataset (which corresponds to a different
                # FashionIQ category)
                
                for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                          idx_to_dress_mapping):
                    if encoder == 'text':
                        index_features, index_names = index_features_list[idx], index_names_list[idx]
                    else:
                        if encoder_arch == "blip2":
                            index_features, index_names = extract_index_features_blip2(classic_val_dataset, blip_visual)
                        else:
                            index_features, index_names = extract_index_features_clip(classic_val_dataset, clip_model)
                    
                    if cir_frame == "sum":
                        if encoder_arch == "blip2":

                            recall_at10, recall_at50 = compute_fiq_val_metrics_blip2(relative_val_dataset,
                                                                       blip_textual,
                                                                       blip_multimodal,
                                                                       index_features,
                                                                       index_names,
                                                                       combining_function)
                        else:
                            recall_at10, recall_at50 = compute_fiq_val_metrics_clip(relative_val_dataset,
                                                                       blip_textual,
                                                                       blip_multimodal,
                                                                       index_features,
                                                                       index_names,
                                                                       combining_function)
                    elif cir_frame == "artemis":
                        recall_at10, recall_at50 = artemis_compute_fiq_val_metrics(relative_val_dataset,
                                                                       blip_textual,
                                                                       blip_multimodal,
                                                                       index_features,
                                                                       index_names,
                                                                       artemis)

                    recalls_at10.append(recall_at10)
                    recalls_at50.append(recall_at50)

                results_dict = {}
                for i in range(len(recalls_at10)):
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict.update({
                    f'average_recall_at10': mean(recalls_at10),
                    f'average_recall_at50': mean(recalls_at50),
                    f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
                })

                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    if encoder_arch == "blip2":
                        save_model('tuned_blip_text_arithmetic', epoch, blip_textual, training_path)
                        save_model('tuned_blip_multi_arithmetic', epoch, blip_multimodal, training_path) 
                        save_model('tuned_blip_visual_arithmetic', epoch, blip_visual, training_path)
                    elif encoder_arch == "clip":
                        save_model('tuned_clip_arithmetic', epoch, clip_model, training_path)
                
                    save_model('tuned_tac', epoch, tac, training_path)
                    save_model('tuned_hca', epoch, hca, training_path)
                if not save_best:
                        print("Warning!!!! Now you don't save any models, please set save_best==True")


def blip_finetune_cirr(num_epochs: int, batch_size: int,
                       validation_frequency: int, transform: str, save_training: bool,  save_best: bool,
                       **kwargs):
    """
    Fine-tune blip on the CIRR dataset using as combining function the image-text element-wise sum
    :param num_epochs: number of epochs
    :param blip_model_name: blip model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['blip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param encoder: which blip encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio`    :return:
    """
    experiment_name = kwargs["experiment_name"]
    # training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/blip_cirr_{experiment_name}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # initialize encoders 
    encoder_arch = kwargs["encoder_arch"]

    # initialize the encoders with different arch
    model_name = kwargs["model_name"]
    if encoder_arch == "blip2":
        # blip2_encoders
        blip_textual = load_model(name=model_name, model_type="pretrain_vitL", is_eval=True, device=device)
        blip_multimodal = load_model(name=model_name, model_type="pretrain_vitL", is_eval=True, device=device)
        blip_visual = load_model(name=model_name, model_type="pretrain_vitL", is_eval=True, device=device)

    elif encoder_arch == "clip":
        clip_model, clip_preprocess = clip.load(model_name, device=device, jit=False)
        clip_model.eval().float()
    # initialize support modules
    embeds_dim = kwargs["embeds_dim"]
    if encoder_arch == "blip2":
        tac = TwinAttentionCompositorBLIP2(embeds_dim).to(device)
        hca = HingebasedCrossAttentionBLIP2(embeds_dim).to(device)
    
    elif encoder_arch == "clip":
        tac = TwinAttentionCompositorCLIP().to(device)
        hca = HingebasedCrossAttentionCLIP(embeds_dim).to(device)
    
    cir_frame = kwargs["cir_frame"]
    artemis = Artemis(embeds_dim).to(device)

    # defined the combining func
    combining_function = element_wise_sum

    # preprocess
    if encoder_arch == "blip2":
        input_dim = 224
    elif encoder_arch == "clip":
        input_dim = clip_model.visual.input_resolution

    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['squarepad', 'targetpad']")

    # Define the validation datasets
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs
    encoder = kwargs["encoder"]

    if encoder_arch == "blip2":
            val_index_features, val_index_names = extract_index_features_blip2(classic_val_dataset, blip_visual)
    if encoder_arch == "clip":
            val_index_features, val_index_names = extract_index_features_clip(classic_val_dataset, clip_model)
    if encoder == 'text':
        if encoder_arch == "blip2":
            val_index_features, val_index_names = extract_index_features_blip2(classic_val_dataset, blip_visual)
        if encoder_arch == "clip":
            val_index_features, val_index_names = extract_index_features_clip(classic_val_dataset, clip_model)
    
    # debug for validation
    if encoder_arch == "blip2":
        results = artemis_compute_cirr_val_metrics(relative_val_dataset,
                                                                   blip_textual,
                                                                   blip_multimodal,
                                                                   val_index_features,
                                                                   val_index_names,
                                                                   artemis)
    else:
        results = compute_cirr_val_metrics_clip(relative_val_dataset, clip_model, val_index_features,
                                                   val_index_names, combining_function)


    # Define the train dataset and the combining function
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=0, pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    learning_rate = kwargs["learning_rate"]
    min_lr = kwargs["min_lr"]
    max_epoch = kwargs["max_epoch"]
    
    # Define the optimizer, the loss and the grad scaler
    if encoder_arch == "blip2": 
        # # blip2_encoder
        if encoder == 'multi': # only finetuning text_encoder
            optimizer = optim.AdamW([  # param in blip_multimodal
                {'params': [param for param in blip_multimodal.Qformer.bert.parameters()], 
                'lr': learning_rate,
                'weight_decay': 0.05},
                {'params': blip_multimodal.query_tokens,
                'lr': learning_rate,
                'weight_decay': 0.05},
                # param in blip_textual
                {'params': [param for param in blip_textual.Qformer.bert.parameters()],
                'lr': learning_rate, 
                'weight_decay': 0.05},

                 # params in support modules
                {'params': [param for param in tac.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},

                {'params': [param for param in hca.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},

                {'params': [param for param in artemis.parameters()], 
                'lr': learning_rate * 10,
                'weight_decay': 0.05},
                ])
                   
        elif encoder == 'both': #  finetuning textual and visual concurrently
            optimizer = optim.AdamW([
                # param in blip_multimodal
                {'params': [param for param in blip_multimodal.Qformer.bert.parameters()], 
                'lr': learning_rate,
                'weight_decay': 0.05},
                {'params': blip_multimodal.query_tokens,
                'lr': learning_rate,
                'weight_decay': 0.05},

                # param in blip_textual
                {'params': [param for param in blip_textual.Qformer.bert.parameters()],
                'lr': learning_rate,
                'weight_decay': 0.05},

                # param in blip_visual
                {'params': [param for param in blip_visual.Qformer.bert.parameters()], 
                'lr': learning_rate,
                'weight_decay': 0.05},
                {'params': blip_visual.query_tokens,
                'lr': learning_rate,
                'weight_decay': 0.05},

                # params in support modules

                {'params': [param for param in tac.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},

                {'params': [param for param in hca.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},

                {'params': [param for param in artemis.parameters()], 
                'lr': learning_rate * 10,
                'weight_decay': 0.05},
                ])

        else:
            raise ValueError("encoders to finetune must be 'multi' or 'both'")
    elif encoder_arch == "clip": 
        # clip_encoder
        optimizer = optim.AdamW([  # param in blip_multimodal
                {'params': [param for name, param in clip_model.named_parameters()
                        if 'visual' not in name], 
                'lr': learning_rate,
                'weight_decay': 0.05},

                 # params in support modules
                {'params': [param for param in tac.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},

                {'params': [param for param in hca.parameters()], 
                'lr': learning_rate * 2,
                'weight_decay': 0.05},
                ])

    # define loss function and scaler
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best results to zero
    if save_best:
        # best_harmonic = 0
        # best_geometric = 0
        best_arithmetic = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # define weights for different modules
    tac_weight = kwargs["tac_weight"]
    hca_weight = kwargs["hca_weight"]

    # epoch loop
    for epoch in range(num_epochs):
        with experiment.train():
            # encoder = "text" or "both"
            # set models to train mode
            if encoder_arch == "blip2":
                # # blip2_encoder
                blip_multimodal.Qformer.bert.train()
                # blip_multimodal.vision_proj.train()
                blip_multimodal.query_tokens.requires_grad = True
                blip_textual.Qformer.bert.train()
                # blip_textual.text_proj.train()

                # both adds param in visual_encoder
                # blip2_encoder
                if encoder == "both":
                    blip_visual.Qformer.bert.train()
                    # blip_visual.vision_proj.train()
                    blip_visual.query_tokens.requires_grad = True
            
            elif encoder_arch == "clip":
                # clip_encoder            
                clip_model.train()

            # support modules
            if tac_weight > 0:
                tac.train()
            if hca_weight > 0:
                hca.train()
            if cir_frame == "artemis":  
                artemis.train()

            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            train_bar = tqdm(relative_train_loader, ncols=150)

            # adjust learning rate in every epoch
            cosine_lr_schedule(optimizer, epoch, max_epoch, learning_rate, min_lr, onlyGroup0=True)
            
            # iteration loop
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):
                images_in_batch = reference_images.size(0)
                step = len(train_bar) * epoch + idx
                optimizer.zero_grad()
                
                # move ref and tar img to device
                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)

                # Extract the features, compute the logits and the loss

                with torch.cuda.amp.autocast():
                    if encoder_arch == "blip2":
                        # text
                        text_embeds = blip_textual.extract_features({"text_input":captions},
                                                                mode="text").text_embeds
                        text_feats = text_embeds[:,0,:]
                        
                        # target
                        target_embeds = blip_visual.extract_features({"image":target_images}, 
                                                                    mode="image").image_embeds
                        target_feats = F.normalize(target_embeds[:,0,:], dim=-1)

                        # reference
                        reference_embeds = blip_multimodal.extract_features({"image":reference_images,
                                                                    "text_input":captions}).multimodal_embeds
                        reference_feats = reference_embeds[:,0,:]
                        
                        # embeds encoded with visual_encoder
                        reference_embeds_for_tac = blip_visual.extract_features({"image":reference_images}, 
                                                                    mode="image").image_embeds
                    elif encoder_arch == "clip":
                        # reference
                        reference_feats, reference_embeds = clip_model.encode_image(reference_images)
                        reference_embeds_for_tac = reference_embeds
                        
                        # text
                        text_inputs = clip.tokenize(captions, context_length=77, truncate=True).to(device,non_blocking=True)
                        text_feats, text_embeds = clip_model.encode_text(text_inputs)
                        
                        # target
                        target_feats, target_embeds = clip_model.encode_image(target_images)
                        target_feats = F.normalize(target_feats)


                # ============ Query-Target Contrastive =========== 
                    
                    if cir_frame == "artemis":
                        # artemis
                        artemis_scores =  artemis.compute_score_broadcast_artemis(reference_feats, text_feats, target_feats)
                        # artemis_logits = artemis.temperature.exp() * artemis_scores
                    elif cir_frame == "sum":
                        # sum_predicted
                        predicted_feats = combining_function(reference_feats, text_feats)
                        matching_logits = 100 * predicted_feats @ target_feats.T

                # ============ Query-Target Align =========== 
                        
                    # align(tac)
                    if tac_weight > 0:
                        visual_gap_feats = tac(reference_embeds_for_tac, target_embeds)
                        aligning_logits = 10 * text_feats @ visual_gap_feats.T

                # ============ Reference-Caption-Target Contrastive =========== 
                    if hca_weight > 0:
                        psudo_T = hca(reference_embeds = reference_embeds,
                                            caption_embeds = text_embeds,
                                            target_embeds = target_embeds)
                        
                        reasoning_logits = 10 * psudo_T @ reference_feats.T 

                # ============ LOSS =========== 
                    # align_loss / tac
                    # align_loss = crossentropy_criterion(align_logits, ground_truth)

                    # hca_loss
                    # hca_tcr_loss = crossentropy_criterion(hca_logits, ground_truth)
                    
                    if cir_frame == "artemis":
                        # artemis_loss
                        # artemis_loss = crossentropy_criterion(artemis_logits, ground_truth)
                        if tac_weight > 0:
                            contrastive_logits = tac_weight * aligning_logits + \
                                            (1 - tac_weight) * artemis_scores
                        else:
                            contrastive_logits = artemis_scores
                    elif cir_frame == "sum":
                        # contrastive loss
                        # contrastive_loss = crossentropy_criterion(contrast_logits, ground_truth)
                        if tac_weight > 0:
                            contrastive_logits = tac_weight * aligning_logits + \
                                            (1 - tac_weight) * matching_logits
                        else:
                            contrastive_logits = matching_logits

                # ========== Sum_Loss  ===============
                    # hca_loss
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    contrastive_loss = crossentropy_criterion(contrastive_logits, ground_truth)
                    if hca_weight > 0:
                        reasoning_loss = crossentropy_criterion(reasoning_logits, ground_truth)
                        loss = hca_weight * reasoning_loss + (1 - hca_weight) * contrastive_loss
                    loss = contrastive_loss

                # Backpropagate and update the weights
                scaler.scale(loss).backward()      
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            with experiment.validate():
                if encoder_arch == "blip2":
                    val_index_features, val_index_names = extract_index_features_blip2(classic_val_dataset, blip_visual)
                    if cir_frame == "sum":
                        results = compute_cirr_val_metrics_blip2(relative_val_dataset,
                                                                 blip_textual,
                                                                 blip_multimodal,
                                                                 val_index_features,
                                                                 val_index_names,
                                                                 combining_function)
                    elif cir_frame == "artemis":
                        results = artemis_compute_cirr_val_metrics(relative_val_dataset,
                                                                   blip_textual,
                                                                   blip_multimodal,
                                                                   val_index_features,
                                                                   val_index_names,
                                                                   artemis)
                elif encoder_arch == "clip":
                    results = compute_cirr_val_metrics_clip(relative_val_dataset, clip_model, val_index_features,
                                                   val_index_names, combining_function)

                            
                group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

                results_dict = {
                    'group_recall_at1': group_recall_at1,
                    'group_recall_at2': group_recall_at2,
                    'group_recall_at3': group_recall_at3,
                    'recall_at1': recall_at1,
                    'recall_at5': recall_at5,
                    'recall_at10': recall_at10,
                    'recall_at50': recall_at50,
                    'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                    'arithmetic_mean': mean(results),
                    'harmonic_mean': harmonic_mean(results),
                    'geometric_mean': geometric_mean(results)
                }
                print(json.dumps(results_dict, indent=4))

                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                if save_training:
                    if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                        best_arithmetic = results_dict['arithmetic_mean']
                        # save encoders
                        if encoder_arch == "blip2":
                            save_model('tuned_blip_text_arithmetic', epoch, blip_textual, training_path)
                            save_model('tuned_blip_multi_arithmetic', epoch, blip_multimodal, training_path) 
                            save_model('tuned_blip_visual_arithmetic', epoch, blip_visual, training_path)
                        elif encoder_arch == "clip":
                            save_model('tuned_clip_arithmetic', epoch, clip_model, training_path)
                        # save support modules anyway
                        save_model('tuned_tac_arithmetic', epoch, tac, training_path)
                        save_model('tuned_hca_arithmetic', epoch, hca, training_path)
                        # save artemis modules
                        if cir_frame == "artemis":
                            save_model('tuned_artemis_arithmetic', epoch, artemis, training_path)
                    if not save_best:
                        print("Warning!!!! Now you don't save any models, please set save_best==True")


if __name__ == '__main__':
    parser = ArgumentParser()
    # dataset
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    # comet enviroment
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    # fine_tune_encoder_modal
    parser.add_argument("--encoder", type=str, default="text", help="the encoder that needs to be finetuned")
    parser.add_argument("--encoder-arch", type=str, default="blip2", help="the encoder architecture")
    parser.add_argument("--model-name", type=str, default="blip2_feature_extractor", help="the model used for encoder  blip2_feature_extractor for blip2 or RN50x4 for clip")
    # training args
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    # cosin learning rate scheduler
    parser.add_argument("--min-lr", default=0, type=float, help="Cos Learning Rate Scheduler min learning rate")
    parser.add_argument("--max-epoch", default=10, type=int, help="Cos Learning Rate Scheduler max epoch")
    #i mage preprocessing
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['blip', 'squarepad', 'targetpad'] ")
    # training settings
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")
    parser.add_argument("--cir-frame", default="sum", type=str, help="frame loss")
    parser.add_argument("--tac-weight", default=0.1, type=float, help="tac_loss weight")
    parser.add_argument("--hca-weight", default=0.1, type=float, help="hca-loss weight")
    parser.add_argument("--hca-temperature", default=0.92, type=float, help="hca_temperature")
    parser.add_argument("--tac-temperature", default=2.3,  type=float, help="tac_temperature")
    parser.add_argument("--embeds-dim", default=768, type=int, help="")

    # fix seed for stable results
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True
    
    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "max_epoch": args.max_epoch,
        "min_lr": args.min_lr,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "experiment_name":args.experiment_name,
        "encoder":args.encoder,
        "encoder_arch":args.encoder_arch,
        "model_name":args.model_name,
        "cir_frame":args.cir_frame,
        "tac_weight": args.tac_weight,
        "hca_weight": args.hca_weight,
        "hca_temperature": args.hca_temperature,
        "tac_temperature": args.tac_temperature,
        "embeds_dim": args.embeds_dim
    }
    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"{args.dataset} blip fine-tuning",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
    else:
        print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    ssl._create_default_https_context = ssl._create_unverified_context

    if args.dataset.lower() == 'cirr':
        blip_finetune_cirr(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        blip_finetune_fiq(**training_hyper_params)
