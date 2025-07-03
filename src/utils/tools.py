import torch
import os
import io
import time
import pandas as pd
import numpy as np
from ast import literal_eval  # For parsing list-like strings
from scipy.stats import entropy
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
import ast

import pickle

from utils.SAC import *


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

def save_results(eval_ids, results, truths, all_vars, mode, output_dir, test):
    # Save predictions
    results_df = pd.DataFrame({
        "ids": eval_ids,
        "Predicted": [list(row.cpu().detach().numpy()) for row in results],  # Continuous outputs before L1 loss
        "Ground_Truth": [list(row.cpu().detach().numpy()) for row in truths],
        "std": [list(row.cpu().detach().numpy()) for row in all_vars]
    })
    output_file = f"{output_dir}_{mode}_results_val.csv"
    results_df.to_csv(output_file, index=False)
    if test:
        results_df.to_csv(f"{output_dir}_{mode}_results.csv", index=False)
    print(f"Saved {mode} predictions to {output_file}")


def kl_divergence_SAC(mean_p, mean_q, sigma_p, sigma_q):
    """
    Computes KL divergence between two Gaussian distributions with learnable variance.

    Args:
        mean_p (float): Mean of unimodal prediction.
        mean_q (float): Mean of multimodal prediction.
        sigma_p (float): sigma of unimodal prediction (learnable).
        sigma_q (float): sigma of multimodal prediction (learnable).

    Returns:
        float: KL divergence.
    """

    kl = np.log(sigma_q / sigma_p) + (sigma_p**2 + (mean_p - mean_q)**2) / (2 * sigma_q**2) - 0.5
    return kl

def compute_cc_weights(unimodal_preds, multimodal_preds):
    """
    Computes Pearson correlation coefficient between unimodal and multimodal predictions.

    Args:
        unimodal_preds (dict): Dictionary of unimodal predictions {modality: np.array of predictions}.
        multimodal_preds (np.array): Multimodal predictions.

    Returns:
        dict: Correlation weights for each modality.
    """
    correlation_weights = {}
    for modality, preds in unimodal_preds.items():
        correlation, _ = pearsonr(preds, multimodal_preds)
        correlation_weights[modality] = correlation if not np.isnan(correlation) else 0  # Handle NaN cases

    return correlation_weights

def compute_mi_weights(unimodal_preds, multimodal_preds):
    """
    Computes Mutual Information between unimodal and multimodal predictions.

    Args:
        unimodal_preds (dict): Dictionary of unimodal predictions {modality: np.array of predictions}.
        multimodal_preds (np.array): Multimodal predictions.

    Returns:
        dict: Mutual information weights for each modality.
    """
    mi_weights = {}
    for modality, preds in unimodal_preds.items():
        mi_score = mutual_info_regression(preds.reshape(-1, 1), multimodal_preds)
        mi_weights[modality] = mi_score[0] if mi_score[0] > 0 else 0  # Ensure non-negative

    return mi_weights

def normalize(x):
    x = np.array(x)
    return (x - np.mean(x)) / (np.std(x) + 1e-6)


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
        if args.kl_type == "residual":
            uni_fold = f"/data/wang/junh/results/MMIM/rebuttal/uni_deberta/"
        elif args.kl_type == "joint":
            uni_fold = f"/data/wang/junh/results/MMIM/unimodal_std/"
        else:
            uni_fold = f"/data/wang/junh/results/MMIM/unimodal_error/"
        unimodal_dfs = []
        for modality in modalities:
            uni_path = os.path.join(uni_fold, f"{args.dataset}_{modality.lower()}_{args.n_class}_{dataset}_results.csv")
            
            if os.path.exists(uni_path):
                df = pd.read_csv(uni_path)

                # Ensure 'Predicted' is parsed correctly from list-like strings
                df['Predicted'] = df['Predicted'].apply(lambda x: literal_eval(x)[0] if isinstance(x, str) else x)
                
                df = df[['ids', 'Predicted', 'std']].rename(columns={'Predicted': modality, 'std': f'sigma_p_{modality}'})
                unimodal_dfs.append(df)
            else:
                print(f"Warning: {uni_path} not found!")
        
        merged_df = unimodal_dfs[0]
        for df in unimodal_dfs[1:]:
            merged_df = merged_df.merge(df, on="ids", how="inner")

        # **2. Load & Match Multimodal Results by `ids`**
        multi_fold = f"/data/wang/junh/results/MMIM/{args.out_folder}/"
        multimodal_file = os.path.join(multi_fold, f"{args.dataset}_text_audio_video_{args.n_class}_{dataset}_results_val.csv")

        multi_df = pd.read_csv(multimodal_file)
        multi_df['Predicted'] = multi_df['Predicted'].apply(lambda x: literal_eval(x)[0] if isinstance(x, str) else x)
        multi_df['std'] = multi_df['std'].apply(
            lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else float(x)
        )

        new_multi = multi_df[['ids', 'Predicted']].rename(columns={'Predicted': 'Multi'})

        merged_df = merged_df.merge(new_multi, on="ids", how="inner")
        merged_df = merged_df.merge(multi_df[['ids', 'std']], on="ids", how="left")

        matched_instances = len(merged_df)
        print(f"Matched {matched_instances} instances between unimodal and multimodal.")

        if args.kl_type == "sac":
            # **3. Train Variance Estimator (Every Time We Update KL Weights)**
            device = torch.device("cuda")
            hidden_values = np.stack(merged_df['hidden'].values)
            input_dim = hidden_values.shape[1]
            hidden_tensor = torch.tensor(hidden_values, dtype=torch.float32, device="cuda")

            pred_means = merged_df['Multi'].values.reshape(-1, 1)
            mean_tensor = torch.tensor(pred_means, dtype=torch.float32, device=device)

            # Step 2: Create dataset and dataloader
            dataset_torch = torch.utils.data.TensorDataset(hidden_tensor, mean_tensor)
            data_loader = torch.utils.data.DataLoader(
                dataset_torch, batch_size=32, shuffle=True,
                generator=torch.Generator(device=device)
            )

            # Initialize variance estimator
            variance_model = VarianceEstimator(input_dim).to(device)

            # Train variance estimator
            variance_model = train_variance_estimator(variance_model, data_loader, device=device)

            # **4. Compute KL-Divergence Weights Using Multimodal Variance**
            variance_model.eval()
            sigma_q = variance_model(hidden_tensor).cpu().detach().numpy().flatten()

        # **Compute KL divergence with learned variance**
        for modality in modalities:
            sigma_p = merged_df[f'sigma_p_{modality}'].apply(lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else x).values.astype(np.float32)
            sigma_p = np.sqrt(sigma_p**2)
            sigma_q = merged_df['std'].values.astype(np.float32)
            sigma_q = np.sqrt(sigma_q**2)
            merged_df[f'kl_{modality}'] = merged_df.apply(
                lambda row: max(0.00001, float(kl_divergence_SAC(row[modality], row['Multi'], sigma_p[row.name], sigma_q[row.name]))),
                axis=1
            )

        if args.weights_type == "global":
            if args.kl_type == "residual":
                global_kl = {}
                for modality in modalities:
                    # Predicted means
                    p = merged_df[modality].values.astype(np.float32)   # modality predictions
                    q = merged_df['Multi'].values.astype(np.float32)    # multimodal predictions

                    # Means of predictions
                    mu_p = np.mean(p)
                    mu_q = np.mean(q)

                    # Variance estimates from averaged squared stds (not from raw residuals)
                    # sigma_p_{modality} is already std (i.e., |y - ŷ|)
                    merged_df[f'sigma_p_{modality}'] = merged_df[f'sigma_p_{modality}'].apply(lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else x).values.astype(np.float32)
                    sigma_p_i = merged_df[f'sigma_p_{modality}'].values.astype(np.float32)
                    merged_df['std'] = merged_df['std'].values.astype(np.float32)
                    sigma_q_i = merged_df['std'].values.astype(np.float32)

                    sigma_p = np.sqrt(np.mean(sigma_p_i ** 2))
                    sigma_q = np.sqrt(np.mean(sigma_q_i ** 2))

                    # Compute global KL divergence
                    kl = kl_divergence_SAC(mu_p, mu_q, sigma_p, sigma_q)
                    global_kl[modality] = kl
            elif args.kl_type == "mi":
                unimodal_preds = {modality: merged_df[modality].values for modality in modalities}
                multimodal_preds = merged_df['Multi'].values
                mi_weights = compute_mi_weights(unimodal_preds, multimodal_preds)
                mi_weights = {modality: max(0.00001, mi_weights[modality]) for modality in modalities}
                # Normalize MI weights
                mi_total = sum(mi_weights.values())
                mi_weights = {k: v / mi_total for k, v in mi_weights.items()}
                merged_df["mi_text"] = mi_weights["text"]
                merged_df["mi_audio"] = mi_weights["audio"]
                merged_df["mi_video"] = mi_weights["video"]
                print(f"Mutual Information Weights: {mi_weights}")

            elif args.kl_type =="cc":
                unimodal_preds = {modality: merged_df[modality].values for modality in modalities}
                multimodal_preds = merged_df['Multi'].values
                cc_weights = compute_cc_weights(unimodal_preds, multimodal_preds)
                cc_weights = {modality: max(0.00001, cc_weights[modality]) for modality in modalities}
                cc_total = sum(cc_weights.values())
                cc_weights = {k: v / cc_total for k, v in cc_weights.items()}
                print(f"Correlation Coefficients: {cc_weights}")
                merged_df["cc_text"] = cc_weights["text"]
                merged_df["cc_audio"] = cc_weights["audio"]
                merged_df["cc_video"] = cc_weights["video"]

        else:      
            # Row-wise normalization: KL weights per instance sum to 1
            kl_cols = [f'kl_{modality}' for modality in modalities]

            # Compute row-wise sum across modalities
            kl_values = merged_df[kl_cols].values  # shape (N, 3)
            row_sums = kl_values.sum(axis=1, keepdims=True)

            # Avoid division by zero
            row_sums[row_sums == 0] = 1e-6

            # Normalize each row
            normalized_kl = kl_values / row_sums

            # Assign back to DataFrame
            for i, modality in enumerate(modalities):
                merged_df[f'kl_{modality}'] = normalized_kl[:, i] 
            
        ##############################################################
        # **Compute Correlation Coefficient**
        unimodal_preds = {modality: merged_df[modality].values for modality in modalities}
        multimodal_preds = merged_df['Multi'].values

        cc_weights = compute_cc_weights(unimodal_preds, multimodal_preds)
        cc_weights = {modality: max(0.00001, cc_weights[modality]) for modality in modalities}
        cc_total = sum(cc_weights.values())
        cc_weights = {k: v / cc_total for k, v in cc_weights.items()}
        print(f"Correlation Coefficients: {cc_weights}")
        merged_df["cc_text"] = cc_weights["text"]
        merged_df["cc_audio"] = cc_weights["audio"]
        merged_df["cc_video"] = cc_weights["video"]

        mi_weights = compute_mi_weights(unimodal_preds, multimodal_preds)
        mi_weights = {modality: max(0.00001, mi_weights[modality]) for modality in modalities}
        # Normalize MI weights
        mi_total = sum(mi_weights.values())
        mi_weights = {k: v / mi_total for k, v in mi_weights.items()}
        merged_df["mi_text"] = mi_weights["text"]
        merged_df["mi_audio"] = mi_weights["audio"]
        merged_df["mi_video"] = mi_weights["video"]
        print(f"Mutual Information Weights: {mi_weights}")

        for modality in modalities:
            # Multiply KL weights by correlation coefficient weights
            #############################################################
            if args.weights_type == "kl":
                merged_df[f'final_weight_{modality}'] = merged_df[f'kl_{modality}']
            elif args.weights_type == "kl+cc": 
                merged_df[f'final_weight_{modality}'] = merged_df[f'kl_{modality}'] * cc_weights[modality]
            elif args.weights_type == "kl+mi":
                merged_df[f'final_weight_{modality}'] = merged_df[f'kl_{modality}'] * mi_weights[modality]
            elif args.weights_type == "global" and args.kl_type == "residual":
                merged_df[f'final_weight_{modality}'] = global_kl[modality]
            elif args.weights_type == "global" and args.kl_type == "mi":
                merged_df[f'final_weight_{modality}'] = mi_weights[modality]
            elif args.weights_type == "global" and args.kl_type == "cc":
                merged_df[f'final_weight_{modality}'] = cc_weights[modality]
        
        # final normalization of multiplied weights sum to 1
        if args.weights_type == "kl+cc" or args.weights_type == "kl+mi":
            final_cols = [f"final_weight_{m}" for m in modalities]
            row_sum = merged_df[final_cols].sum(axis=1)
            merged_df[final_cols] = merged_df[final_cols].div(row_sum, axis=0)
        
        # **LOG ALL WEIGHTS**
        log_path = os.path.join(f"/data/wang/junh/results/MMIM/{args.out_folder}/", f"{args.dataset}_weights_log_epoch{epoch}.csv")

        # Save to file
        merged_df.to_csv(log_path, index=False)

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
                    ###############################################################
                    weight_list[-3 + j] = (smooth_factor * matching_row[f'final_weight_{modality}']) + (1 - smooth_factor) * weight_list[-3 + j]
                    # weight_list[-3 + j] = matching_row[f'final_weight_{modality}']

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