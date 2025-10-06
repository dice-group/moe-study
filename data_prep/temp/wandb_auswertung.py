import wandb

api = wandb.Api()
runs = api.runs("dice-nlp/llama32_multilingual_eval")
include_base_keys = [
    "include_base_44_albanian", "include_base_44_arabic", "include_base_44_armenian",
    "include_base_44_azerbaijani", "include_base_44_basque", "include_base_44_belarusian",
    "include_base_44_bengali", "include_base_44_bulgarian", "include_base_44_chinese",
    "include_base_44_croatian", "include_base_44_dutch", "include_base_44_estonian",
    "include_base_44_finnish", "include_base_44_french", "include_base_44_georgian",
    "include_base_44_german", "include_base_44_greek", "include_base_44_hebrew",
    "include_base_44_hindi", "include_base_44_hungarian", "include_base_44_indonesian",
    "include_base_44_italian", "include_base_44_japanese", "include_base_44_kazakh",
    "include_base_44_korean", "include_base_44_lithuanian", "include_base_44_malay",
    "include_base_44_malayalam", "include_base_44_nepali", "include_base_44_north macedonian",
    "include_base_44_persian", "include_base_44_polish", "include_base_44_portuguese",
    "include_base_44_russian", "include_base_44_serbian", "include_base_44_spanish",
    "include_base_44_tagalog", "include_base_44_tamil", "include_base_44_telugu",
    "include_base_44_turkish", "include_base_44_ukrainian", "include_base_44_urdu",
    "include_base_44_uzbek", "include_base_44_vietnamese"
]

xnli_keys = [
    "xnli_ar", "xnli_bg", "xnli_de", "xnli_el", "xnli_en", "xnli_es", "xnli_fr",
    "xnli_hi", "xnli_ru", "xnli_sw", "xnli_th", "xnli_tr", "xnli_ur", "xnli_vi", "xnli_zh"
]

tasks = {"include_base": include_base_keys, "xnli": xnli_keys}

summary = runs[0].summary._json_dict

print("\n### ACCURACY SUMMARY ###")
for task, keys in tasks.items():
    print(f"\n## {task.upper()} ##")
    for key in keys:
        val = summary.get(key)
        if val is not None:
            print(f"  {key:30s} {val:.4f}")
        else:
            print(f"  {key:30s} (not found)")
