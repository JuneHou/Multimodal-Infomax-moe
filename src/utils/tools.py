import torch
import os
import io
import time
import pandas as pd
import numpy as np
from ast import literal_eval  # For parsing list-like strings
from scipy.stats import entropy
import pickle


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Generate timestamp
    name = f"{args.dataset}_{args.modality}_{args.n_class}_{args.lr_main}_{args.d_vh}_{args.d_vout}"  # Construct filename

    if not os.path.exists('pre_trained_models'):
        os.mkdir('pre_trained_models')

    torch.save(model.state_dict(), f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    # name = save_load_name(args, name)
    name = f"{args.dataset}_{args.modality}_{args.n_class}_{args.lr_main}_{args.d_vh}_{args.d_vout}"
    with open(f'pre_trained_models/{name}.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    model = torch.load(buffer)
    return model


def random_shuffle(tensor, dim=0):
    if dim != 0:
        perm = (i for i in range(len(tensor.size())))
        perm[0] = dim
        perm[dim] = 0
        tensor = tensor.permute(perm)
    
    idx = torch.randperm(t.size(0))
    t = tensor[idx]

    if dim != 0:
        t = t.permute(perm)
    
    return t

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_results(eval_ids, results, truths, mode, output_dir):
    # Save predictions
    results_df = pd.DataFrame({
        "ids": eval_ids,
        "Predicted": [list(row.cpu().detach().numpy()) for row in results],  # Continuous outputs before L1 loss
        "Ground_Truth": [list(row.cpu().detach().numpy()) for row in truths]
    })
    output_file = f"{output_dir}_{mode}_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Saved {mode} predictions to {output_file}")

def kl_divergence_gaussian(mean_p, mean_q, sigma=0.1):
    """
    Computes KL divergence between two Gaussian distributions N(mean_p, sigma^2) and N(mean_q, sigma^2).
    """
    kl = np.log(sigma / sigma) + (sigma**2 + (mean_p - mean_q)**2) / (2 * sigma**2) - 0.5
    return kl

def update_kl_weights(args, epoch, smooth_factor, datasets, new_weights_path):
    """
    Updates KL divergence-based modality weights and saves new `.pkl` files.
    
    Args:
        args: Training arguments containing file paths.
        epoch (int): Current training epoch.
        smooth_factor (float): Factor for weight updates.
        datasets (list): Dataset splits to update (e.g., ['train', 'val', 'test']).
    """
    modalities = ['text', 'audio', 'video']
    for dataset in datasets:
        print(f"Updating weights for {dataset} dataset...")

        # **1. Merge Unimodal Results by `ids`**
        uni_fold = f"/data/wang/junh/results/MMIM/unimodal/"
        unimodal_dfs = []
        for modality in modalities:
            uni_path = os.path.join(uni_fold, f"{args.dataset}_{modality.lower()}_{args.n_class}_{dataset}_results.csv")
            
            if os.path.exists(uni_path):
                df = pd.read_csv(uni_path)

                # Ensure 'Predicted' is parsed correctly from list-like strings
                df['Predicted'] = df['Predicted'].apply(lambda x: literal_eval(x)[0] if isinstance(x, str) else x)
                
                df = df[['ids', 'Predicted']].rename(columns={'Predicted': modality})
                unimodal_dfs.append(df)
            else:
                print(f"Warning: {uni_path} not found!")
        
        merged_df = unimodal_dfs[0]
        for df in unimodal_dfs[1:]:
            merged_df = merged_df.merge(df, on="ids", how="inner")

        print(f"Merged {len(merged_df)} unimodal instances.")

        # **2. Load & Match Multimodal Results by `ids`**
        multi_fold = f"/data/wang/junh/results/MMIM/{args.out_folder}/"
        multimodal_file = os.path.join(multi_fold, f"{args.dataset}_text_audio_video_{args.n_class}_{dataset}_results.csv")

        multi_df = pd.read_csv(multimodal_file)
        multi_df['Predicted'] = multi_df['Predicted'].apply(lambda x: literal_eval(x)[0] if isinstance(x, str) else x)
        multi_df = multi_df[['ids', 'Predicted']].rename(columns={'Predicted': 'Multi'})

        merged_df = merged_df.merge(multi_df, on="ids", how="inner")

        matched_instances = len(merged_df)
        print(f"Matched {matched_instances} instances between unimodal and multimodal.")

        # **3. Compute KL-Divergence Weights (Gaussian Approximation)**
        sigma = 0.1  # Small variance to define Gaussians
        for modality in modalities:
            merged_df[f'kl_{modality}'] = merged_df.apply(
                lambda row: kl_divergence_gaussian(row[modality], row['Multi'], sigma), axis=1
            )

        # Normalize KL scores across instances
        for modality in modalities:
            max_kl = merged_df[f'kl_{modality}'].max()
            min_kl = merged_df[f'kl_{modality}'].min()
            if max_kl - min_kl > 0:
                merged_df[f'kl_{modality}'] = (merged_df[f'kl_{modality}'] - min_kl) / (max_kl - min_kl)
            else:
                merged_df[f'kl_{modality}'] = 0.001  # Prevent zero division

        # **4. Update Weights in Pickle File (Moving Average)**
        if "new_weights" in args.dataset_path:
            pkl_file = os.path.join(args.dataset_path, f"{dataset}.pkl")
        else:
            pkl_file = os.path.join(args.dataset_path, f"{args.dataset.upper()}/{dataset}.pkl")

        with open(pkl_file, 'rb') as file:
            stays_list = pickle.load(file)

        match_count = 0
        
        for i, stay in enumerate(stays_list):
            stay_id = stay[-1]  # Extract the ID from the last element

            if stay_id in merged_df['ids'].astype(str).values:
                matching_row = merged_df[merged_df['ids'].astype(str) == stay_id].iloc[0]

                # Convert the tuple to a list for modification
                stay_list = list(stay)

                # Convert stay[-3] (which is a tuple) to a list for modification
                weight_list = list(stay_list[-3])  

                # Compute new weights using a moving average
                for j, modality in enumerate(modalities):
                    weight_list[-3 + j] = (smooth_factor * matching_row[f'kl_{modality}']) + (1 - smooth_factor) * weight_list[-3 + j]

                # Convert updated weights back to a tuple
                stay_list[-3] = tuple(weight_list)

                # Update stays_list at index i
                stays_list[i] = tuple(stay_list)  # Assign the modified tuple back

                match_count += 1

        print(f"Updated {match_count} out of {len(stays_list)} records.")

        # **5. Save Updated Pickle File**
        updated_pkl_file = os.path.join(new_weights_path, f"{dataset}.pkl")
        # Check if the file exists; if not, create it
        if not os.path.exists(updated_pkl_file):
            os.makedirs(os.path.dirname(updated_pkl_file), exist_ok=True)
            print(f"{updated_pkl_file} does not exist. Creating a new one.")

        # Save the updated stays_list to a new pickle file
        with open(updated_pkl_file, 'wb') as file:
            pickle.dump(stays_list, file)

        print(f"Saved updated weights to {updated_pkl_file}")

    print(f"Weight updates complete for {datasets}.")