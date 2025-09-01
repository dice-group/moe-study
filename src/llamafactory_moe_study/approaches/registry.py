from llamafactory_moe_study.approaches.adamole.adapter import AdaMoLEAdapter

_REGISTRY = {
    "adamole": AdaMoLEAdapter(),
    "adamoe_1": "", #tbd
    "adamoe_2": "", #tbd
}

def get_selected_method(name: str):
    return _REGISTRY.get(name.lower())