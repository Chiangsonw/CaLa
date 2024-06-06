import multiprocessing
import random
from pathlib import Path
from typing import Union, Tuple, List

import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import CIRRDataset, FashionIQDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def extract_features_fusion_blip(blip_textual, multimodal_embeds, captions):
    self = blip_textual.Qformer.bert
    embeddings = self.embeddings
    encoder = self.encoder
    # pooler = blip_textual.Qformer.bert.pooler

    text_tokens = blip_textual.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

    text_atts = text_tokens.attention_mask
    query_atts = torch.ones(multimodal_embeds.size()[:-1], dtype=torch.long).to(device)
    # print("query_atts:", query_atts.shape)
    # print("text_atts:", text_atts.shape)
    attention_mask = torch.cat([query_atts, text_atts], dim=1)
    # print("attention_mask:",attention_mask.shape)
    # head_mask = blip_textual.Qformer.bert.get_head_mask(head_mask,
    #             blip_textual.Qformer.bert.config.num_hidden_layers)

    embedding_output = embeddings(
            input_ids=text_tokens.input_ids,
            query_embeds=multimodal_embeds,)
    
    input_shape = embedding_output.size()[:-1]
    extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape, device, False
            )
    # print(extended_attention_mask.shape)
    head_mask = self.get_head_mask(None, self.config.num_hidden_layers)

    encoder_outputs = encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            return_dict=True,
        )
    sequence_output = encoder_outputs[0]
    return sequence_output


# def extract_index_features_blip(dataset: Union[CIRRDataset, FashionIQDataset], model: CLIP, vision_layer) -> \
def extract_index_features_blip2(dataset: Union[CIRRDataset, FashionIQDataset], blip_model) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param model: CLIP model
    :return: a tensor of features and a list of images
    """
    feature_dim = 768
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0,feature_dim)).to(device, non_blocking=True)
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = blip_model.extract_features({"image":images}, mode="image").image_embeds[:,0,:]
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)
    return index_features, index_names

def extract_index_features_blip1(dataset: Union[CIRRDataset, FashionIQDataset], blip_model) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param model: CLIP model
    :return: a tensor of features and a list of images
    """
    feature_dim = 256
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0,feature_dim)).to(device, non_blocking=True)
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = blip_model(images)[:,0,:]
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)
    return index_features, index_names

def extract_index_features_blip_feature_extractor(dataset: Union[CIRRDataset, FashionIQDataset], model) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param model: CLIP model
    :return: a tensor of features and a list of images
    """
    # feature_dim = model.visual.output_dim
    feature_dim = 256
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0,feature_dim)).to(device, non_blocking=True)
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            # visual_encoder = model.visual_encoder.float()
            # batch_vision_embeds = visual_encoder(images)
            # batch_vision_embeds = model.ln_vision(batch_vision_embeds).float()
            # batch_features = model.extract_features({"image":images}, mode="image").image_embeds
            
            batch_features = model.extract_features({"image":images}, mode="image").image_embeds_proj[:,0,:]
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)
    return index_features, index_names

def extract_index_features_clip(dataset: Union[CIRRDataset, FashionIQDataset], clip_model: CLIP) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param clip_model: CLIP model
    :return: a tensor of features and a list of images
    """
    feature_dim = clip_model.visual.output_dim
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)[0]
            # print(clip_model.encode_image(images)[1].shape)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)
    return index_features, index_names

def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions
def generate_randomized_fiq_caption_blip(flattened_captions: List[str],txt_processors:callable) -> List[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        caption =''
        if random_num < 0.25:
            caption=f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}"
        elif 0.25 < random_num < 0.5:
            caption=f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}"
        elif 0.5 < random_num < 0.75:
            caption=f"{flattened_captions[i].strip('.?, ').capitalize()}"
        else:
            caption=f"{flattened_captions[i + 1].strip('.?, ').capitalize()}"
        captions.append(txt_processors(caption))
    return captions

def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu',
                                                               non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss: {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path):
    """
    Save the weights of the model during training
    :param name: name of the file
    :param cur_epoch: current epoch
    :param model_to_save: pytorch model to be saved
    :param training_path: path associated with the training run
    """
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, str(models_path / f'{name}.pt'))

import math
def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr, onlyGroup0=False):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    param_group_count = 0
    for param_group in optimizer.param_groups:
        param_group_count += 1
        if param_group_count <= 1 and onlyGroup0: # only vary group0 parameters' learning rate, i.e., exclude the text_proj layer
            param_group['lr'] = lr

@torch.no_grad()        
def _momentum_update(model_pairs, momentum):
    for model_pair in model_pairs:           
        for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
            param_m.data = param_m.data * momentum + param.data * (1. - momentum)

@torch.no_grad()
def _dequeue_and_enqueue(model, target_feats, idx, queue_size):
    # gather keys before updating queue
    batch_size = target_feats.shape[0]

    ptr = int(model.queue_ptr)
    assert queue_size % batch_size == 0  # for simplicity

    # replace the keys at ptr (dequeue and enqueue)
    model.target_queue[:, ptr:ptr + batch_size] = target_feats.T
    model.idx_queue[:, ptr:ptr + batch_size] = idx.T
    ptr = (ptr + batch_size) % queue_size  # move pointer

    model.queue_ptr[0] = ptr


def l2norm(x):
	"""L2-normalize each row of x"""
	norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
	return torch.div(x, norm)
