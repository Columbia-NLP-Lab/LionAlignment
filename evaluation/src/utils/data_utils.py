import hashlib


def add_full_id(data_dict: dict):
    text_chosen = data_dict['chosen']
    text_rejected = data_dict['rejected']
    full_encoded = f"{text_chosen} {text_rejected}"
    full_encoded_id = hashlib.sha256(full_encoded.encode("utf-8")).hexdigest()
    data_dict['full_id'] = full_encoded_id
    return data_dict