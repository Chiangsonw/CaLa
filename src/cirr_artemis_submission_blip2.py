import json
import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import CIRRDataset_val_submission, targetpad_transform, squarepad_transform, base_path
from utils import element_wise_sum, device, extract_index_features_blip2
from lavis.models import load_model
from artemis import Artemis


def generate_cirr_test_submissions(artemis,  file_name: str, blip_textual, blip_multimodal, 
                                   blip_visual, preprocess: callable):
    """
   Generate and save CIRR test submission files to be submitted to evaluation server
   :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                           features
   :param file_name: file_name of the submission
   :param clip_model: CLIP model
   :param preprocess: preprocess pipeline
   """
    blip_textual = blip_textual.float().eval()
    blip_multimodal = blip_multimodal.float().eval()
    blip_visual = blip_visual.float().eval()
    artemis = artemis.float().eval()

    # Define the dataset and extract index features
    classic_test_dataset = CIRRDataset_val_submission('test1', 'classic', preprocess)
    index_features, index_names = extract_index_features_blip2(classic_test_dataset, blip_visual)
    relative_test_dataset = CIRRDataset_val_submission('test1', 'relative', preprocess)

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(relative_test_dataset, blip_textual, blip_multimodal,
                                                                                  index_features, index_names,
                                                                                  artemis)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define submission path
    submissions_folder_path = base_path / "submission" / 'CIRR'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving CIRR test predictions")
    with open(submissions_folder_path / f"retrieval_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"retrieval_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_dicts(relative_test_dataset: CIRRDataset_val_submission, blip_textual, blip_multimodal, index_features: torch.tensor,
                             index_names: List[str], artemis) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute test prediction dicts for CIRR dataset
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: test index features
    :param index_names: test index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """

    # Generate predictions
    artemis_scores, reference_names, group_members, pairs_id = \
        generate_cirr_test_predictions(blip_textual, blip_multimodal, relative_test_dataset, artemis, index_names,
                                       index_features)

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - artemis_scores
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts for test split
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                             zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                   zip(pairs_id, sorted_group_names)}
    
    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(blip_textual, blip_multimodal, relative_test_dataset: CIRRDataset_val_submission, artemis,
                                   index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[List[str]], List[str]]:
    """
    Compute CIRR predictions on the test set
    :param clip_model: CLIP model
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                           features
    :param index_features: test index features
    :param index_names: test index names

    :return: predicted_features, reference_names, group_members and pairs_id
    """
    print(f"Compute CIRR test predictions")

    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32,
                                      num_workers=multiprocessing.cpu_count(), pin_memory=True)

    # Get a mapping from index names to index features
    # name_to_feat = dict(zip(index_names, index_features))

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    artemis_scores = torch.empty((0, len(index_names))).to(device, non_blocking=True)
    group_members = []
    reference_names = []

    for batch_pairs_id, batch_reference_names, reference_images, captions, batch_group_members in tqdm(
            relative_test_loader):  # Load data
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
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)
        # target_hard_names.extend(target_hard_name)
        # captions_list.extend(captions)
        
    return artemis_scores, reference_names, group_members, pairs_id
    #return artemis_scores, reference_names, group_members, pairs_id, target_hard_names, captions_list


def main():
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, required=True, help="submission file name")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--blip2-textual-path", type=str, help="Path to the fine-tuned BLIP2 model")
    parser.add_argument("--blip2-visual-path", type=str, help="Path to the fine-tuned BLIP2 model")
    parser.add_argument("--blip2-multimodal-path", type=str, help="Path to the fine-tuned BLIP2 model")
    parser.add_argument("--artemis-path", type=str, help="Path to the fine-tuned BLIP2 model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    args = parser.parse_args()
    
    blip_textual_path = args.blip2_textual_path
    blip_visual_path =  args.blip2_visual_path
    blip_multimodal_path = args.blip2_multimodal_path
    artemis_path = args.artemis_path

    blip_textual = load_model(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
    x_saved_state_dict = torch.load(blip_textual_path, map_location=device)

    blip_textual.load_state_dict(x_saved_state_dict["Blip2Qformer"])

    blip_multimodal = load_model(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
    r_saved_state_dict = torch.load(blip_multimodal_path, map_location=device)
    blip_multimodal.load_state_dict(r_saved_state_dict["Blip2Qformer"])

    blip_visual = load_model(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
    t_saved_state_dict = torch.load(blip_visual_path, map_location=device)
    blip_visual.load_state_dict(t_saved_state_dict["Blip2Qformer"])

    artemis = Artemis(768).to(device)
    a_saved_state_dict = torch.load(artemis_path, map_location=device)
    artemis.load_state_dict(a_saved_state_dict["Artemis"])


    input_dim = 224
    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)

    if args.combining_function.lower() == 'sum':
        combining_function = element_wise_sum
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    generate_cirr_test_submissions(artemis, args.submission_name, blip_textual, blip_multimodal, blip_visual, preprocess)


if __name__ == '__main__':
    main()
