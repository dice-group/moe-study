from approaches.adamole.adapter import AdaMoLEAdapter
from approaches.mola.adapter import MolaAdapter

_REGISTRY = {
    "adamole": AdaMoLEAdapter(),
    "mola": MolaAdapter(),
    "adamoe_2": "", #tbd
}

def get_selected_method(name: str):
    return _REGISTRY.get(name.lower())