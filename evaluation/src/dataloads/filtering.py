def filter_distilabel_orca_dpo_pairs(data_dict):
    if data_dict['other_info']['in_gsm8k_train']:
        return False
    return True

## register the filtering functions. Most of them does NOOP.
FILTERING_FN = {
    "argilla/ultrafeedback-binarized-preferences-cleaned": lambda _: True,
    "argilla/dpo-mix-7k":  lambda _: True,
    "HuggingFaceH4/ultrafeedback_binarized":  lambda _: True,
    "HuggingFaceH4/ultrachat_200k":  lambda _: True,
    "when2rl/distilabel-intel-orca-dpo-pairs_cleaned_reformatted": filter_distilabel_orca_dpo_pairs
}