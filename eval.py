def get_rank_score_mat_list(load_path, model_list, task_idx, gpu, p_total_user, p_total_item, 
                            total_train_dataset, total_valid_dataset, total_test_dataset, rank_importance,
                            new_user_train_mat, new_user_valid_mat, new_user_test_mat, k_list, eval_task_idx):
    """ 
    Get the rank score matrix for an ensemble teacher.
    
    Args:
        load_path (str): Path to load the model files.
        model_list (list): List of model names to be evaluated.
        task_idx (int): Index of the current task.
        gpu (int): GPU id to load the models.
        p_total_user (int): Total number of users to consider.
        p_total_item (int): Total number of items to consider.
        total_train_dataset, total_valid_dataset, total_test_dataset: Datasets for training, validation, and testing.
        rank_importance: Rank importance value for evaluation.
        new_user_train_mat, new_user_valid_mat, new_user_test_mat: Data matrices for new users.
        k_list (list): List of k values for evaluation metrics (e.g., Recall@k).
        eval_task_idx (int): Index for evaluation task.

    Returns:
        rank_score_mat_list (list): List of rank score matrices for each model.
    """

    # Initialize an empty list to hold rank score matrices for each model
    rank_score_mat_list = []
    
    # Iterate over each model in the given model list
    for m_name in model_list:
        
        # Construct the model file path based on the load_path, model name, and task index
        model_path = os.path.join(load_path, m_name, f"TASK_{task_idx}.pth")
        print(f"model_name = {m_name} and model_path = {model_path}")
        
        # Load the model's score matrix from the specified path
        score_mat = torch.load(model_path, map_location=torch.device(gpu))["score_mat"]
        
        # Slice the score matrix to match the specified number of users and items
        score_mat = score_mat[:p_total_user, :p_total_item]
        
        # Get the top 1000 items (indices) for each user, and convert to numpy array
        sorted_mat = to_np(torch.topk(score_mat, k=1000).indices.detach().cpu())
        
        # Evaluate Collaborative Learning (CL) results using the total datasets
        get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, sorted_mat, k_list, eval_task_idx)

        # Evaluation for new users
        print(f"\t[The Result of new users in {task_idx}-th Block]")
        new_user_results = get_eval_with_mat(new_user_train_mat, new_user_valid_mat, new_user_test_mat, sorted_mat, k_list)
        print(f"\tvalid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}")
        
        # Convert the score matrix to a rank score matrix using rank importance and append to the list
        rank_score_mat = convert_to_rank_score_mat(score_mat, rank_importance)
        rank_score_mat_list.append(rank_score_mat.detach().cpu())
        
    # Return the list of rank score matrices for all the models
    return rank_score_mat_list
def get_eval_Ensemble(E_score_mat, total_train_dataset, total_valid_dataset, total_test_dataset, task_idx, k_list, 
             new_user_train_mat, new_user_valid_mat, new_user_test_mat, print_out = "Teacher"):
    """
    Evaluate the ensemble score matrix for the specified datasets.
    
    Args:
        E_score_mat: Ensemble score matrix.
        total_train_dataset, total_valid_dataset, total_test_dataset: Datasets for training, validation, and testing.
        task_idx (int): Index of the current task.
        k_list (list): List of k values for evaluation metrics (e.g., Recall@k).
        new_user_train_mat, new_user_valid_mat, new_user_test_mat: Data matrices for new users.
        print_out (str): Optional label for printing results.
    
    """
    # Convert the score matrix to a sorted matrix of top items
    E_sorted_mat = score2sorted(E_score_mat)
    print(f"\n[Ensemble for {print_out}]")
    
    # Evaluate Collaborative Learning (CL) results using the total datasets
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, E_sorted_mat, k_list, task_idx)
    
    # Evaluation for new users
    print(f"\t[The Result of new users in {task_idx}-th Block]")
    new_user_results = get_eval_with_mat(new_user_train_mat, new_user_valid_mat, new_user_test_mat, E_sorted_mat, k_list)
    print(f"\tvalid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}")

def get_eval(model, gpu, train_loader, test_dataset, k_list):
    """
    Evaluate a single model on the given datasets.
    
    Args:
        model: The model to evaluate.
        gpu (int): GPU id to use for evaluation.
        train_loader: DataLoader for training data.
        test_dataset: Dataset containing validation and test matrices.
        k_list (list): List of k values for evaluation metrics (e.g., Recall@k).
    
    Returns:
        eval_results (dict): Evaluation results for validation and test sets.
        score_mat (Tensor): Score matrix of user-item predictions.
        sorted_mat (ndarray): Sorted item indices based on scores.
    """
    # Extract training, validation, and test matrices
    train_mat = train_loader.dataset.rating_mat
    valid_mat = test_dataset.valid_mat
    test_mat = test_dataset.test_mat
    
    # Define the maximum k value for evaluation
    max_k = max(k_list)
    eval_results = create_metrics(k_list)
    
    # Calculate the score matrix based on the model's similarity type
    if model.sim_type == "inner product":
        user_emb, item_emb = model.get_embedding()
        score_mat = torch.matmul(user_emb, item_emb.T)
    
    elif model.sim_type == "UAE":
        score_mat = torch.zeros(model.user_count, model.item_count)
        for mini_batch in train_loader:
            mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}
            output = model.forward_eval(mini_batch)
            score_mat[mini_batch['user'], :] = output.cpu()
            
    # Convert score matrix to sorted item indices
    score_mat = score_mat.detach().cpu()
    sorted_mat = torch.topk(score_mat, k=1000, dim=-1, largest=True).indices
    sorted_mat = to_np(sorted_mat)

    # Evaluate results for each user in the test matrix
    for test_user in test_mat:
        sorted_list = list(sorted_mat[test_user])

        for mode in ["valid", 'test']:
            sorted_list_tmp = []
            
            # Define ground truth and items already seen based on the mode
            if mode == "valid":
                gt_mat = valid_mat
                already_seen_items = set(train_mat[test_user]) | set(test_mat[test_user].keys())

            elif mode == "test":
                gt_mat = test_mat
                already_seen_items = set(train_mat[test_user]) | set(valid_mat[test_user].keys())

            # Filter out already seen items from the sorted list
            for item in sorted_list:
                if item not in already_seen_items:
                    sorted_list_tmp.append(item)
                if len(sorted_list_tmp) > max_k: 
                    break
            
            # Calculate evaluation metrics for different k values
            for k in k_list:
                hit_k = len(set(sorted_list_tmp[:k]) & set(gt_mat[test_user].keys()))

                # Precision & Recall
                eval_results[mode][f"P{k}"].append(hit_k / min(k, len(gt_mat[test_user].keys())))
                eval_results[mode][f"R{k}"].append(hit_k / len(gt_mat[test_user].keys()))

                # NDCG
                denom = np.log2(np.arange(2, k + 2))
                dcg_k = np.sum(np.in1d(sorted_list_tmp[:k], list(gt_mat[test_user].keys())) / denom)
                idcg_k = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), k)])
                NDCG_k = dcg_k / idcg_k
                
                eval_results[mode][f"N{k}"].append(NDCG_k)

    # Calculate the average results for each mode and k value
    for mode in ["valid", "test"]:
        for k in k_list:
            eval_results[mode][f"P{k}"] = round(np.mean(eval_results[mode][f"P{k}"]), 4)
            eval_results[mode][f"R{k}"] = round(np.mean(eval_results[mode][f"R{k}"]), 4)
            eval_results[mode][f"N{k}"] = round(np.mean(eval_results[mode][f"N{k}"]), 4)
    
    return eval_results, score_mat, sorted_mat

def get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, sorted_mat, k_list, current_task_idx, 
                  FB_flag=False, return_value=False, max_task=5):
    """
    Get Collaborative Learning (CL) results for the given task index and sorted matrix.
    
    Args:
        total_train_dataset, total_valid_dataset, total_test_dataset: Datasets for training, validation, and testing.
        sorted_mat: Sorted matrix of user-item scores.
        k_list (list): List of k values for evaluation metrics.
        current_task_idx (int): Index of the current task.
        FB_flag (bool): Whether to use Feedback (FB) training. Default is False.
        return_value (bool): Whether to return the evaluation results. Default is False.
        max_task (int): Maximum number of tasks to evaluate. Default is 5.
    
    Returns:
        If return_value is True, returns lists of validation and test results for each task.
    """
    if return_value:
        valid_list = []
        test_list = []
        
    # Iterate over all tasks up to the current task or max_task
    for before_task_id in range(min(current_task_idx + 1, max_task + 1)):
        if before_task_id > 0 and FB_flag:
            # Merge train dictionaries if Feedback training is enabled
            train_dict = merge_train_dict(train_dict, total_train_dataset[f"TASK_{before_task_id}"])
        else:
            train_dict = total_train_dataset[f"TASK_{before_task_id}"]
            
        # Create rating matrices for training, validation, and test datasets
        train_mat = make_rating_mat(train_dict)
        valid_mat = make_rating_mat(total_valid_dataset[f"TASK_{before_task_id}"])
        test_mat = make_rating_mat(total_test_dataset[f"TASK_{before_task_id}"])
        
        # Get evaluation results using the matrices
        results = {}
        results = get_eval_with_mat(train_mat, valid_mat, test_mat, sorted_mat, k_list)
        
        # Print results for the current task
        print(f"\n[TASK_ID:{before_task_id}]")
        valid_dict = {f"valid_{key}": value for key, value in results["valid"].items()}
        test_dict = {f"test_{key}": value for key, value in results["test"].items()}
        print(valid_dict)
        print(test_dict)
        
        # Append the results to lists if return_value is True
        if return_value:
            valid_list.append(valid_dict)
            test_list.append(test_dict)
    
    # Return lists of validation and test results if return_value is True
    if return_value:
        return valid_list, test_list
    

def get_eval_with_mat(train_mat, valid_mat, test_mat, sorted_mat, k_list, target_users=None):
    """
    Evaluate user-item interaction matrices with given sorted prediction matrix.
    
    Args:
        train_mat: Training user-item interaction matrix.
        valid_mat: Validation user-item interaction matrix.
        test_mat: Test user-item interaction matrix.
        sorted_mat: Sorted prediction matrix (user-item ranking).
        k_list (list): List of k values for evaluation metrics.
        target_users (list, optional): List of target users to evaluate. If None, evaluates all users in test_mat.
    
    Returns:
        eval_results (dict): Dictionary containing precision, recall, and NDCG values for validation and test sets.
    """
    max_k = max(k_list)
    eval_results = create_metrics(k_list)

    # Determine the users to evaluate
    if target_users is not None:
        test_users = target_users
    else:
        test_users = list(test_mat.keys())

    # Evaluate each user in the list of test users
    for test_user in test_users:
        try:
            sorted_list = list(sorted_mat[test_user])
        except:
            continue

        for mode in ["valid", 'test']:
            sorted_list_tmp = []
            
            # Determine the ground truth and items already seen based on the evaluation mode
            try:
                if mode == "valid":
                    gt_mat = valid_mat
                    already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())

                elif mode == "test":
                    gt_mat = test_mat
                    already_seen_items = set(train_mat[test_user].keys()) | set(valid_mat[test_user].keys())
            except:
                continue
            
            # Filter out already seen items from the sorted list
            for item in sorted_list:
                if item not in already_seen_items:
                    sorted_list_tmp.append(item)
                if len(sorted_list_tmp) > max_k:
                    break

            # Calculate evaluation metrics for different k values
            for k in k_list:
                hit_k = len(set(sorted_list_tmp[:k]) & set(gt_mat[test_user].keys()))

                # Precision & Recall
                eval_results[mode][f"P{k}"].append(hit_k / min(k, len(gt_mat[test_user].keys())))
                eval_results[mode][f"R{k}"].append(hit_k / len(gt_mat[test_user].keys()))

                # NDCG
                denom = np.log2(np.arange(2, k + 2))
                dcg_k = np.sum(np.in1d(sorted_list_tmp[:k], list(gt_mat[test_user].keys())) / denom)
                idcg_k = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), k)])
                NDCG_k = dcg_k / idcg_k
                
                eval_results[mode][f"N{k}"].append(NDCG_k)

    # Calculate the average results for each mode and k value
    for mode in ["valid", "test"]:
        for k in k_list:
            eval_results[mode][f"P{k}"] = round(np.asarray(eval_results[mode][f"P{k}"]).mean(), 4)
            eval_results[mode][f"R{k}"] = round(np.asarray(eval_results[mode][f"R{k}"]).mean(), 4)
            eval_results[mode][f"N{k}"] = round(np.asarray(eval_results[mode][f"N{k}"]).mean(), 4)
    
    return eval_results
