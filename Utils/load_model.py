import torch


def load_teacher_model(model_path):
    model_dict = torch.load(model_path)
    return model_dict


def load_teacher_ckpt(model_path):
    model_dict = torch.load(model_path)
    return model_dict["checkpoint"]


def load_teacher_score_mat(model_path):
    model_dict = torch.load(model_path)
    return model_dict["score_mat"]


def load_teacher_sorted_mat(model_path):
    model_dict = torch.load(model_path)
    return model_dict["sorted_mat"]
