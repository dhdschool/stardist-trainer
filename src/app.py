import panel as pn
import sys
#import widgets
from pathlib import Path

pn.extension('mathjax', design="material", sizing_mode="stretch_width")

def exit(event):
    if event:
        sys.exit()

def exclusive(target, event):
    target.value = not event.new

sidebar_widgth = 300

image_dir_widget = pn.widgets.FileInput(directory=True)
image_text_widget = pn.widgets.StaticText(name="Image Directory")
image_widget = pn.Column(image_text_widget, image_dir_widget)

model_text_widget = pn.widgets.StaticText(name="Model Directory")
model_dir_widget = pn.widgets.FileInput(directory=True)
model_widget = pn.Column(model_text_widget, model_dir_widget)

filedir = pn.Row(image_widget, model_widget)


exit_widget = pn.widgets.Button(name='Server Shutdown', button_type='danger')
train_widget = pn.widgets.Button(name='Train Model', button_type='primary')


with open(Path('md') / Path('info.md')) as file:
    info = file.read()
info_widget = pn.pane.Markdown(info)

pn.bind(exit, exit_widget, watch=True)

vgg_switch = pn.widgets.Switch(value=False, width=40)
vgg_text = pn.widgets.StaticText(name='VGG VIA Format')
vgg_widget = pn.layout.Row(vgg_text, vgg_switch, sizing_mode='fixed', width=sidebar_widgth)

ij_switch = pn.widgets.Switch(value=True, width=40)
ij_text = pn.widgets.StaticText(name='ImageJ Format')
ij_widget = pn.layout.Row(ij_text, ij_switch, sizing_mode='fixed', width=sidebar_widgth)

model_ow_switch = pn.widgets.Switch(value=False, width=40)
model_ow_text = pn.widgets.StaticText(name='Overwrite Model Config')
model_ow_widget = pn.Row(model_ow_text, model_ow_switch, sizing_mode='fixed', width=sidebar_widgth)

ij_switch.link(vgg_switch, callbacks={'value':exclusive})
vgg_switch.link(ij_switch, callbacks={'value':exclusive})


val_input = pn.widgets.IntInput(value=20, start=1, end=100)
val_text = pn.widgets.StaticText(name="Validation split percentage")
val_widget = pn.Row(val_text, val_input, sizing_mode='fixed', width=sidebar_widgth)

epoch_input = pn.widgets.IntInput(value=1, start=1)
epoch_text = pn.widgets.StaticText(name="Epochs")
epoch_widget = pn.Row(epoch_text, epoch_input, sizing_mode='fixed', width=sidebar_widgth)

coord_formats = [
    "XYC",
    "YXC"
]

image_format_text = pn.widgets.StaticText(name="Image coord format")
image_format_selector = pn.widgets.Select(options=coord_formats)
image_format_widget = pn.Row(image_format_text, image_format_selector, sizing_mode='fixed', width=sidebar_widgth)

mask_format_text = pn.widgets.StaticText(name="Mask coord format")
mask_format_selector = pn.widgets.Select(options=coord_formats)
mask_format_widget = pn.Row(mask_format_text, mask_format_selector, sizing_mode='fixed', width=sidebar_widgth)


toggles = pn.Column(ij_widget,
                    vgg_widget,
                    model_ow_widget)

hyperparameters = pn.Column(epoch_widget,
                            val_widget,
                            image_format_widget,
                            mask_format_widget)

mainbar = pn.Column(filedir,
                    train_widget, 
                    exit_widget)

sidebar = pn.Column(info_widget, 
                    pn.Spacer(width=200),
                    hyperparameters, 
                    toggles)

app = pn.template.MaterialTemplate(
    title='Stardist Trainer',
    main=mainbar,
    sidebar=sidebar
).servable()

if __name__ == 'main__':
    pn.serve({"app.py":app}, port=5006)