import hashlib


def reformat_argilla_ultrafeedback(data_dict: dict):
    out_data_dict = {}
    out_data_dict['prompt'] = data_dict['prompt']
    out_data_dict['prompt_id'] = hashlib.sha256(data_dict['prompt'].encode("utf-8")).hexdigest()
    out_data_dict['chosen'] = data_dict['chosen']
    out_data_dict['rejected'] = data_dict['rejected']
    out_data_dict['messages'] = data_dict['chosen']
    out_data_dict['score_chosen'] = data_dict['chosen-rating']
    out_data_dict['score_rejected'] = data_dict['rejected-rating']
    out_data_dict['other_info'] = {
        'source': data_dict['source'],
        'chosen-model': data_dict['chosen-model'],
        'rejected-model': data_dict['rejected-model'],
    }
    return out_data_dict


def reformat_argilla_dpo_mix(data_dict: dict):
    out_data_dict = {}
    out_data_dict['prompt'] = data_dict['chosen'][0]['content']
    out_data_dict['prompt_id'] = hashlib.sha256(out_data_dict['prompt'].encode("utf-8")).hexdigest()
    out_data_dict['chosen'] = data_dict['chosen']
    out_data_dict['rejected'] = data_dict['rejected']
    out_data_dict['messages'] = data_dict['chosen']
    out_data_dict['score_chosen'] = data_dict['chosen_rating']
    out_data_dict['score_rejected'] = data_dict['rejected_rating']
    out_data_dict['other_info'] = {
        'source': data_dict['dataset'],
    }
    return out_data_dict


def reformat_ultrafeedback_binarized(data_dict: dict):
    data_dict['other_info'] = {}
    return data_dict


def reformat_ultrachat(data_dict: dict):
    data_dict['other_info'] = {}
    return data_dict


def reformat_wandb_deita_10k(data_dict: dict):
    data_dict['other_info'] = {}
    return data_dict


## register the reformatting function
REFORMATTING_FN = {
    "argilla/ultrafeedback-binarized-preferences-cleaned": reformat_argilla_ultrafeedback,
    "argilla/dpo-mix-7k": reformat_argilla_dpo_mix,
    "HuggingFaceH4/ultrafeedback_binarized": reformat_ultrafeedback_binarized,
    "HuggingFaceH4/ultrachat_200k": reformat_ultrachat,
    "wandb/deita-10k-v0-sft-latin": reformat_wandb_deita_10k
}