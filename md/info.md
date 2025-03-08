### Info
In ImageJ format, the image directory you select must contain two subdirectories named **images** and **masks**. An image in the **images** directory should have a corresponding mask with the same name in the **masks** directory. 

In VGG VIA format, the image directory you select must contain two subdirectories named **images** and **csv**. An image in the **images** directory should have a corresponding csv file with the same name in the **csv** directory.

In either format, the model directory you select should be **either** the folder that contains an existing stardist model's **.config** file **or** the directory you want to save a new model. If you want to save a new model, or overwrite an existing one, select the **Overwrite Model Config** setting. Please be aware that this will delete any existing model weights contained in that directory.

---