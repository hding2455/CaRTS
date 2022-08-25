from .base_render import BaseRender
render_dict = {"BaseRender":BaseRender}

def build_render(render, device):
    return render_dict[render['name']](render['params'], device)
