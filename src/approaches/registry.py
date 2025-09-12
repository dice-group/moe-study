from approaches.adamole.adapter import AdaMoLEAdapter
from approaches.adamoe_1.adapter import MolaAdapter

_REGISTRY = {
    "adamole": AdaMoLEAdapter(),
    "mola": MolaAdapter(),
    "moelpr": "", #tbd
}

def get_selected_method(name: str):
    return _REGISTRY.get(name.lower())