from approaches.adamole.adapter import AdaMoLEAdapter
from approaches.adamoe_1.adapter import MolaAdapter
from approaches.moelpr.adapter import MoelprAdapter

_REGISTRY = {
    "adamole": AdaMoLEAdapter(),
    "mola": MolaAdapter(),
    "moelpr": MoelprAdapter(),
}

def get_selected_method(name: str):
    return _REGISTRY.get(name.lower())
