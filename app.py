#!/usr/bin/env python

import os

import gradio as gr
import torch

from app_image_to_3d import create_demo as create_demo_image_to_3d
from app_text_to_3d import create_demo as create_demo_text_to_3d
from model import Model

DESCRIPTION = '# [Shap-E](https://github.com/openai/shap-e)'

if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value='Duplicate Space for private use',
                       elem_id='duplicate-button',
                       visible=os.getenv('SHOW_DUPLICATE_BUTTON') == '1')
    with gr.Tabs():
        with gr.Tab(label='Text to 3D'):
            create_demo_text_to_3d(model)
        with gr.Tab(label='Image to 3D'):
            create_demo_image_to_3d(model)
demo.queue(max_size=10).launch()
