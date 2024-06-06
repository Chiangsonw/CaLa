import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from utils import extract_index_features_clip, collate_fn, element_wise_sum, device , extract_index_features_blip2
from lavis.models import load_model


def compute_fiq_val_metrics_blip2(relative_val_dataset: FashionIQDataset, blip_textual,blip_multimodal, index_features: torch.tensor,
                            index_names: List[str], combining_function: callable) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    # Generate predictions
    predicted_features, target_names = generate_fiq_val_predictions_blip2(blip_textual, blip_multimodal, relative_val_dataset,
                                                                    combining_function, index_names, index_features)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50

def compute_fiq_val_metrics_clip(relative_val_dataset: FashionIQDataset, clip_model: CLIP, index_features: torch.tensor,
                            index_names: List[str], combining_function: callable) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    # Generate predictions
    predicted_features, target_names = generate_fiq_val_predictions_clip(clip_model, relative_val_dataset,
                                                                    combining_function, index_names, index_features)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def artemis_compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, blip_textual,blip_multimodal, index_features: torch.tensor,
                            index_names: List[str], artemis) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()
    print("Compute FashionIQ validation predictions")
    
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn)
    feature_dim = 768
    # Initialize predicted features, target_names, group_members and reference_names
    artemis.eval()
    artemis_scores = torch.empty((0, len(index_names))).to(device, non_blocking=True)
    target_names = []
    reference_names = []

    for reference_images, batch_reference_names, batch_target_names, captions in tqdm(
            relative_val_loader):  # Load data
        reference_images = reference_images.to(device)

        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}"
            for i in range(0, len(flattened_captions), 2)]

        # Compute the predicted features
        with torch.no_grad():
            text_feats = blip_textual.extract_features({"text_input":input_captions},
                                                               mode="text").text_embeds[:,0,:]
            reference_feats = blip_multimodal.extract_features({"image":reference_images,
                                                                        "text_input":input_captions}).multimodal_embeds[:,0,:]
            batch_artemis_score = torch.empty((0, len(index_names))).to(device, non_blocking=True)
            for i in range(reference_feats.shape[0]):
                one_artemis_score = artemis.compute_score_artemis(reference_feats[i].unsqueeze(0), text_feats[i].unsqueeze(0), index_features)
                batch_artemis_score = torch.vstack((batch_artemis_score, one_artemis_score))
        artemis_scores = torch.vstack((artemis_scores, F.normalize(batch_artemis_score, dim=-1)))
        target_names.extend(batch_target_names)
        reference_names.extend(batch_reference_names) 

    # Compute the distances and sort the results
    distances = 1 - artemis_scores
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions_blip2(blip_textual, blip_multimodal, relative_val_dataset: FashionIQDataset,
                                 combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    # name_to_feat = dict(zip(index_names, index_features))
    feature_dim = 768
    # Initialize predicted features and target names
    predicted_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    target_names = []

    for reference_images, batch_target_names, captions in tqdm(relative_val_loader):  # Load data
        reference_images = reference_images.to(device)
        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}"
            for i in range(0, len(flattened_captions), 2)]

        # Compute the predicted features
        with torch.no_grad():
            text_feats = blip_textual.extract_features({"text_input":input_captions},
                                                         mode="text").text_embeds[:,0,:]
            reference_feats = blip_multimodal.extract_features({"image":reference_images,
                                                                 "text_input":input_captions}).multimodal_embeds[:,0,:]
            
            batch_predicted_features = combining_function(reference_feats, text_feats)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)

    return predicted_features, target_names


def generate_fiq_val_predictions_clip(clip_model: CLIP, relative_val_dataset: FashionIQDataset,
                                 combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []

    for reference_image, reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        text_inputs = clip.tokenize(input_captions, context_length=77).to(device, non_blocking=True)

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)[0]
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)

    return predicted_features, target_names


def fashioniq_val_retrieval(dress_type: str, combining_function: callable, clip_model: CLIP, preprocess: callable):
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)

    return compute_fiq_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                   combining_function)

def artemis_compute_cirr_val_metrics(relative_val_dataset: CIRRDataset, blip_textual, blip_multimodal, index_features: torch.tensor,
                             index_names: List[str], artemis) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    
    print("Compute CIRR validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn)
    feature_dim = 768
    # Initialize predicted features, target_names, group_members and reference_names
    artemis.eval()
    
    artemis_scores = torch.empty((0, len(index_names))).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for reference_images, batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        reference_images = reference_images.to(device)
        batch_group_members = np.array(batch_group_members).T.tolist()
        # Compute the predicted features
        with torch.no_grad():
            text_feats = blip_textual.extract_features({"text_input":captions},
                                                               mode="text").text_embeds[:,0,:]
            reference_feats = blip_multimodal.extract_features({"image":reference_images,
                                                                        "text_input":captions}).multimodal_embeds[:,0,:]
            batch_artemis_score = torch.empty((0, len(index_names))).to(device, non_blocking=True)
            for i in range(reference_feats.shape[0]):
                one_artemis_score = artemis.compute_score_artemis(reference_feats[i].unsqueeze(0), text_feats[i].unsqueeze(0), index_features)
                batch_artemis_score = torch.vstack((batch_artemis_score, one_artemis_score))
        
        artemis_scores = torch.vstack((artemis_scores, F.normalize(batch_artemis_score, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)   


    # Compute the distances and sort the results
    distances = 1 - artemis_scores
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50

def compute_cirr_val_metrics_clip(relative_val_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = \
        generate_cirr_val_predictions_clip(clip_model, relative_val_dataset, combining_function, index_names, index_features)

    print("Compute CIRR validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50

def generate_cirr_val_predictions_clip(clip_model: CLIP, relative_val_dataset: CIRRDataset,
                                  combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)[0]
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members


def compute_cirr_val_metrics_blip2(relative_val_dataset: CIRRDataset, blip_textual, blip_multimodal, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = \
        generate_cirr_val_predictions_blip2(blip_textual, blip_multimodal, relative_val_dataset, combining_function, index_names, index_features)

    print("Compute CIRR validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions_blip2(blip_textual, blip_multimodal, relative_val_dataset: CIRRDataset,
                                  combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    # name_to_feat = dict(zip(index_names, index_features))

    feature_dim = 768
    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for reference_images, batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        reference_images = reference_images.to(device)
        batch_group_members = np.array(batch_group_members).T.tolist()
        # Compute the predicted features
        with torch.no_grad():
            text_feats = blip_textual.extract_features({"text_input":captions},
                                                               mode="text").text_embeds[:,0,:]
            reference_feats = blip_multimodal.extract_features({"image":reference_images,
                                                                        "text_input":captions}).multimodal_embeds[:,0,:]
            
            batch_predicted_features = combining_function(text_feats, reference_feats)
        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members


def compute_cirr_val_metrics_blip1(relative_val_dataset: CIRRDataset, blip_textual, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = \
        generate_cirr_val_predictions_blip1(blip_textual, relative_val_dataset, combining_function, index_names, index_features)

    print("Compute CIRR validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions_blip1(blip_textual, relative_val_dataset: CIRRDataset,
                                  combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    feature_dim = 256
    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_images, batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        
        batch_group_members = np.array(batch_group_members).T.tolist()
        # Compute the predicted features
        with torch.no_grad():
            text_features = blip_textual(captions, max_length=77, device=device)[:,0,:]
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members




def cirr_val_retrieval(combining_function: callable, blip_visual, blip_textual, blip_multimodal, preprocess):
    """
    Perform retrieval on CIRR validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    blip_visual = blip_visual.float().eval()
    blip_textual = blip_textual.float().eval()
    blip_multimodal = blip_multimodal.float().eval()
    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    index_features, index_names = extract_index_features_blip2(classic_val_dataset, blip_visual)
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)

    return compute_cirr_val_metrics_blip2(relative_val_dataset, blip_textual, blip_multimodal,  index_features, index_names,
                                    combining_function)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    # parser.add_argument("--combiner-path", type=Path, help="path to trained Combiner")
    # parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    # parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    # parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    # parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    
    parser.add_argument("--blip2-textual-path", type=str, help="Path to the fine-tuned BLIP2 model")
    parser.add_argument("--blip2-visual-path", type=str, help="Path to the fine-tuned BLIP2 model")
    parser.add_argument("--blip2-multimodal-path", type=str, help="Path to the fine-tuned BLIP2 model")

    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")


    args = parser.parse_args()

    # clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    # input_dim = clip_model.visual.input_resolution
    # feature_dim = clip_model.visual.output_dim

    blip_textual_path = args.blip2_textual_path
    blip_visual_path =  args.blip2_visual_path
    blip_multimodal_path = args.blip2_multimodal_path

    blip_textual = load_model(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
    x_saved_state_dict = torch.load(blip_textual_path, map_location=device)
    # print(x_saved_state_dict.keys())
    blip_textual.load_state_dict(x_saved_state_dict["Blip2Qformer"])

    blip_multimodal = load_model(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
    r_saved_state_dict = torch.load(blip_multimodal_path, map_location=device)
    blip_multimodal.load_state_dict(r_saved_state_dict["Blip2Qformer"])

    blip_visual = load_model(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
    t_saved_state_dict = torch.load(blip_visual_path, map_location=device)
    blip_visual.load_state_dict(t_saved_state_dict["Blip2Qformer"])


    # blip_textual = load_model(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)

    # blip_multimodal = load_model(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)

    # blip_visual = load_model(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)

    # if args.clip_model_path:
    #     print('Trying to load the CLIP model')
    #     saved_state_dict = torch.load(args.clip_model_path, map_location=device)
    #     clip_model.load_state_dict(saved_state_dict["CLIP"])
    #     print('CLIP model loaded successfully')
    input_dim = 224
    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    # else:
    #     print('CLIP default preprocess pipeline is used')
    #     preprocess = clip_preprocess

    if args.combining_function.lower() == 'sum':
        combining_function = element_wise_sum
    # elif args.combining_function.lower() == 'combiner':
    #     combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
    #     state_dict = torch.load(args.combiner_path, map_location=device)
    #     combiner.load_state_dict(state_dict["Combiner"])
    #     combiner.eval()
    #     combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    if args.dataset.lower() == 'cirr':
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(combining_function, blip_visual, blip_textual, blip_multimodal , preprocess)

        print(f"{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")

    elif args.dataset.lower() == 'fashioniq':
        average_recall10_list = []
        average_recall50_list = []

        shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval('shirt', combining_function, clip_model,
                                                                     preprocess)
        average_recall10_list.append(shirt_recallat10)
        average_recall50_list.append(shirt_recallat50)

        dress_recallat10, dress_recallat50 = fashioniq_val_retrieval('dress', combining_function, clip_model,
                                                                     preprocess)
        average_recall10_list.append(dress_recallat10)
        average_recall50_list.append(dress_recallat50)

        toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval('toptee', combining_function, clip_model,
                                                                       preprocess)
        average_recall10_list.append(toptee_recallat10)
        average_recall50_list.append(toptee_recallat50)

        print(f"\n{shirt_recallat10 = }")
        print(f"{shirt_recallat50 = }")

        print(f"{dress_recallat10 = }")
        print(f"{dress_recallat50 = }")

        print(f"{toptee_recallat10 = }")
        print(f"{toptee_recallat50 = }")

        print(f"average recall10 = {mean(average_recall10_list)}")
        print(f"average recall50 = {mean(average_recall50_list)}")
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


if __name__ == '__main__':
    main()
