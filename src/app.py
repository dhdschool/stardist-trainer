import panel as pn
import sys
from model import StarDistAPI
#import widgets
from pathlib import Path

pn.extension('mathjax', design="material", sizing_mode="stretch_width")

def exit(event):
    if event:
        sys.exit()

def train(event):
    if event:
        img_fp = image_dir_widget.value
        model_fp = model_dir_widget.value
        if img_fp is None or model_fp is None:
            return
        
        img_fp = img_fp[0]
        model_fp = model_fp[0]
        
        image_dir = Path(img_fp)
        model_dir = Path(model_fp)
        
        epochs = epoch_input.value
        batch_size = batch_input.value
        validation_percentage = val_input.value
        image_format = image_format_selector.value
        mask_format = mask_format_selector.value
        imagej = ij_switch.value
        overwrite = model_ow_switch.value
        
        classes = None if co_classes_input.value == 1 else co_classes_input.value
        config = {
            'axes': co_axes_selector.value[:2],
            'n_rays': co_nrays_input.value,
            'n_channel_in': co_channels_input.value,
            'n_classes': classes
        }
        
        model = StarDistAPI(image_dir,
                            model_dir,
                            epochs,
                            batch_size,
                            validation_percentage,
                            image_format,
                            mask_format,
                            imagej,
                            overwrite,
                            config_kwargs=config
                            )
        
        train_gen = model.train()
        loadingbar_widget.active = True
        for progress in train_gen:
            loadingbar_widget.value = int(round(progress * 100))
        loadingbar_widget.active = False
        

def exclusive(target, event):
    target.value = not event.new

sidebar_widgth = 300

image_dir_widget = pn.widgets.FileSelector(file_pattern='', root_directory=str(Path.home()), only_files=False)
image_text_widget = pn.widgets.StaticText(name="Image Directory")
image_widget = pn.Column(image_text_widget, image_dir_widget)

model_text_widget = pn.widgets.StaticText(name="Model Directory")
model_dir_widget = pn.widgets.FileSelector(file_pattern='', root_directory=str(Path.home()), only_files=False)
model_widget = pn.Column(model_text_widget, model_dir_widget)

filedir = pn.Column(image_widget, pn.Spacer(height=50), model_widget)


exit_widget = pn.widgets.Button(name='Server Shutdown', button_type='danger')
train_widget = pn.widgets.Button(name='Train Model', button_type='primary')
loadingbar_widget = pn.widgets.Progress(max=100, bar_color='success')

with open(Path('md') / Path('info.md')) as file:
    info = file.read()
info_widget = pn.pane.Markdown(info)

pn.bind(exit, exit_widget, watch=True)
pn.bind(train, train_widget, watch=True)

vgg_switch = pn.widgets.Switch(value=False, width=40)
vgg_text = pn.widgets.StaticText(name='VGG VIA Format', align='center')
vgg_widget = pn.layout.Row(vgg_text, vgg_switch, sizing_mode='fixed', width=sidebar_widgth)

ij_switch = pn.widgets.Switch(value=True, width=40)
ij_text = pn.widgets.StaticText(name='ImageJ Format', align='center')
ij_widget = pn.layout.Row(ij_text, ij_switch, sizing_mode='fixed', width=sidebar_widgth)

model_ow_switch = pn.widgets.Switch(value=False, width=40)
model_ow_text = pn.widgets.StaticText(name='Overwrite Model Config', align='center')
model_ow_widget = pn.Row(model_ow_text, model_ow_switch, sizing_mode='fixed', width=sidebar_widgth)

ij_switch.link(vgg_switch, callbacks={'value':exclusive})
vgg_switch.link(ij_switch, callbacks={'value':exclusive})


val_input = pn.widgets.IntInput(value=20, start=1, end=100)
val_text = pn.widgets.StaticText(name="Validation split percentage", align='center')
val_widget = pn.Row(val_text, val_input, sizing_mode='fixed', width=sidebar_widgth)

epoch_input = pn.widgets.IntInput(value=1, start=1)
epoch_text = pn.widgets.StaticText(name="Epochs", align='center')
epoch_widget = pn.Row(epoch_text, epoch_input, sizing_mode='fixed', width=sidebar_widgth)

batch_input = pn.widgets.IntInput(value=4, start=1)
batch_text = pn.widgets.StaticText(name="Batch Size", align='center')
batch_widget = pn.Row(batch_text, batch_input, sizing_mode='fixed', width=sidebar_widgth)

coord_formats = [
    "YXC",
    "XYC"
]

image_format_text = pn.widgets.StaticText(name="Image coord format", align='center')
image_format_selector = pn.widgets.Select(options=coord_formats)
image_format_widget = pn.Row(image_format_text, image_format_selector, sizing_mode='fixed', width=sidebar_widgth)

mask_format_text = pn.widgets.StaticText(name="Mask coord format", align='center')
mask_format_selector = pn.widgets.Select(options=coord_formats)
mask_format_widget = pn.Row(mask_format_text, mask_format_selector, sizing_mode='fixed', width=sidebar_widgth)

config_overwrite_width = 200
co_axes_text = pn.widgets.StaticText(name="axes", align='center')
co_axes_selector = pn.widgets.Select(options=coord_formats)
co_axes_widget = pn.Row(co_axes_text, co_axes_selector, sizing_mode='fixed', width=config_overwrite_width)

co_nrays_text = pn.widgets.StaticText(name="n_rays", align='center')
co_nrays_input = pn.widgets.IntInput(value=32, start=1)
co_nrays_widget = pn.Row(co_nrays_text, co_nrays_input, sizing_mode='fixed', width=config_overwrite_width)

co_channels_text = pn.widgets.StaticText(name="n_channels_in", align='center')
co_channels_input = pn.widgets.IntInput(value=3, start=1)
co_channels_widget = pn.Row(co_channels_text, co_channels_input, sizing_mode='fixed', width=config_overwrite_width)

co_classes_text = pn.widgets.StaticText(name="n_classes", align='center')
co_classes_input = pn.widgets.IntInput(value=1, start=1)
co_classes_widget = pn.Row(co_classes_text, co_classes_input, sizing_mode='fixed', width=config_overwrite_width)

config_overwrite_data = pn.layout.Column(
    co_axes_widget,
    co_nrays_widget,
    co_channels_widget,
    co_classes_widget
)

config_overwrite_dropdown = pn.layout.Accordion(("Config Overwrite Dropdown", config_overwrite_data))


toggles = pn.Column(ij_widget,
                    vgg_widget,
                    model_ow_widget)

hyperparameters = pn.Column(epoch_widget,
                            batch_widget,
                            val_widget,
                            image_format_widget,
                            mask_format_widget)

mainbar = pn.Column(filedir,
                    pn.Spacer(height=20),
                    loadingbar_widget,
                    train_widget, 
                    exit_widget)

sidebar = pn.Column(info_widget, 
                    pn.Spacer(height=20),
                    hyperparameters, 
                    toggles,
                    pn.Spacer(height=20),
                    config_overwrite_dropdown,
                    pn.Spacer(height=50))

app = pn.template.MaterialTemplate(
    title='Stardist Trainer',
    main=mainbar,
    sidebar=sidebar
).servable()

if __name__ == 'main__':
    pn.serve({"app.py":app}, port=5006)