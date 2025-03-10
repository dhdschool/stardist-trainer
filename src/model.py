from stardist.models import StarDist2D, Config2D
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import os
import json
from matplotlib import pyplot as plt
from csbdeep.utils import normalize
#TODO: ImageJ loader
class StarDistAPI:
    def __init__(self,
                 image_dir,
                 model_dir,
                 epochs=1,
                 batch_size=4,
                 val_per=20,
                 image_format='XYC',
                 mask_format='XYC',
                 imagej=True,
                 overwrite=False,
                 config_kwargs={},
                ):
        
        image_dir = Path(image_dir).absolute()
        self.image_dir = Path(image_dir) / Path('images')
        self.mask_dir = Path(image_dir) / Path('masks')
        self.csv_dir = Path(image_dir) / Path('csv')
        
        model_dir = Path(model_dir).absolute()
        model_name = model_dir.name
        model_dir = model_dir.parent
        
        config = Config2D(**config_kwargs) if overwrite else None
        
        self.model = StarDist2D(config=config,
                                name=model_name,
                                basedir=model_dir)
        
        if config == None:
            self.thresholds_optimized = False
        else:
            self.thresholds_optimized = True
         
        self.epochs = epochs
        self.val_size = int(round((val_per / 100) * batch_size, 0))
        self.image_format = image_format
        self.mask_format = mask_format
        self.imagej = imagej
        self.batch_size = batch_size
        
        
    def train(self):
        if self.imagej:
            loader = ImageJLoader(self.image_dir, self.mask_dir)
        if not self.imagej:
            loader = VGGLoader(self.image_dir, self.csv_dir)
            
        length = len(loader)
        dataset = loader.data()

        for epoch in range(self.epochs):    
            X_train, y_train = [], []
            X_val, y_val = [], []
            for X, y in dataset:
                if len(X_val) < self.val_size:
                    X_val.append(X), y_val.append(y)
                else:
                    X_train.append(X)
                    y_train.append(y)
                
                if len(X_train) == self.batch_size:
                    X_arr = np.array(X_train)
                    X_val_arr = np.array(X_val)
                    
                    if self.model.config.axes != self.image_format:
                        X_arr = X_arr.transpose(axes=(1, 0, 2))
                        X_val_arr = X_val_arr.transpose(axes=(1, 0, 2))

                    y_arr = np.array(y_train)
                    y_val_arr = np.array(y_val)
                    
                    if self.model.config.axes != self.mask_format:
                        y_arr = y_arr.transpose(axes=(1, 0, 2))
                        y_val_arr = y_val_arr.transpose(axes=(1, 0, 2))
                    
                    if not self.thresholds_optimized:
                        self.model.optimize_thresholds(X_val_arr, y_val_arr)
                        self.thresholds_optimized = True
                    
                    X_train = []; y_train = []
                    X_val = []; y_val = []
                    
                    class_labels_dct = [{i+1:i} for i in range(self.model.config.n_classes)]
                    
                    self.model.train(X_arr, y_arr, validation_data=(X_val_arr, y_val_arr, class_labels_dct), epochs=1, classes=class_labels_dct)
                    yield (length * (epoch) + (length - len(loader))) / length * self.epochs # Percentage of training done
class Loader:
    def __init__(self,
                 image_dir: os.PathLike,
                 label_dir: os.PathLike,
                 memory_limit:int=32):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.memory_limit = memory_limit
        
        self.image_paths = list(map(lambda x: Path(str(self.image_dir / Path(x))),
                                    os.listdir(self.image_dir)))
        
        self.label_paths =  list(map(lambda x: Path(str(self.label_dir / Path(x))),
                                    os.listdir(self.label_dir)))
        
        self.image_len = (0, min(self.memory_limit, len(self.image_paths)))
        self.length = len(self.image_paths)
        
        self.images = {}
        self.labels = {}
        
        self._load_images()
    
    def _load_images(self):
        if self.image_len[0] >= self.length:
            self.image_len = (0, min(self.memory_limit, self.image_paths))
        
        for index in range(*self.image_len):
            name = self.image_paths[index].stem
            path = self.image_paths[index]
            
            with Image.open(path) as img:
                img_rgb = img.convert('RGB')
                img_rgb = np.array(img_rgb)
                img_rgb = normalize(img_rgb, 1, 99.8)
                self.images[name] = img_rgb
                self._load_label(name, img.size)
            
    def _load_label(self, name, size):
       pass
    
    def data(self):
        for index in range(len(self.image_paths)):
            name = self.image_paths[index].stem
            
            if len(self.images) == 0 and index != len(self.image_paths) - 1:
                self.image_len = (self.image_len[1]+1, min(self.image_len[1] + self.memory_limit, len(self.image_paths) - self.image_len[1]))
                self._load_images()
            
            img, lbl = self.images.pop(name), self.labels.pop(name)
            self.length -= 1
            
           
            yield img, lbl
    
    def __len__(self):
        return self.length
    
 
class ImageJLoader(Loader):
    def __init__(self,
                 image_dir: os.PathLike,
                 mask_dir: os.PathLike,
                 memory_limit: int=32):
        
        super().__init__(image_dir=image_dir, label_dir=mask_dir, memory_limit=memory_limit)
        
    #Overwrite
    def _load_label(self, name, *args):
        path = list(self.label_dir.glob(f"{name}.*"))[0]
        with Image.open(path) as label:
            label = np.array(label)
            self.labels[name] = label
        
class VGGLoader(Loader):
    def __init__(self,
                 image_dir: os.PathLike,
                 csv_dir: os.PathLike,
                 memory_limit:int=32):
        super().__init__(image_dir=image_dir, label_dir=csv_dir, memory_limit=memory_limit)
        

    def _load_label(self, name, size):
        path = self.label_dir / Path(name + ".csv")
        self.labels[name] = self._csv_to_label_mask(path, size)
        
    
    def _csv_to_label_mask(self, path, img_size):
        def add_label(row):
            shape = json.loads(row['region_shape_attributes'])
            classes = json.loads(row['region_attributes'])
            class_int = int(classes['class_name']) + 1
            
            if shape['name'] == 'circle':
                canvas.circle(xy=(shape['cx'], shape['cy']), radius=shape['r'], fill=class_int, outline=class_int)
            else:
                raise ValueError('Shape not a circle, please implement non circular shapes in this code')
        
        mask = Image.new(mode='L', size=img_size, color=0)
        canvas = ImageDraw.Draw(mask, mode='L')
        data = pd.read_csv(path)[['region_shape_attributes', 'region_attributes']]
        data.apply(add_label, axis=1)
                
        return np.array(mask)
    
if __name__ == '__main__':
    loader = VGGLoader('data/images', 'data/csv')
    

# if __name__ == '__main__':
    
#     train = False
    
#     if train:
#         model = StarDist2D(config=Config2D(
#                 n_channel_in=3
#             ), name='empty-model', basedir='test/model')
#         model.prepare_for_training()

#         X_train = []
#         y_train = []
#         for _ in range(10):
#             X, y = next(dataset)
#             X_train.append(X)
#             y_train.append(y)
            
#         X_val = []
#         y_val = []
#         for _ in range(5):
#             X, y = next(dataset)
#             X_val.append(X)
#             y_val.append(y)
        
#     else:
#         model = StarDist2D(config=None, name='empty-model', basedir='test/model')
   
#     loader.give_counts = True
#     X_test = []
#     y_test = []
#     for _ in range(5):
#         X, y = next(dataset)
#         X_test.append(X)
#         y_test.append(y)
    
#     if train:
#         loader.give_counts = False
#         X_train = np.array(X_train)
#         y_train = np.array(y_train)
        
#         X_val = np.array(X_val)
#         y_val = np.array(y_val)    
        
#         print('Model is training...')
#         history = model.train(X_train, y_train, validation_data=(X_val, y_val), epochs=1)
#         #model.
#         print('Model done training!')
    
       
#     X_test = np.array(X_test)
#     y_test = np.array(y_test)
    
    
#     print('Model is predicting...')
#     for img in X_test:
#         mask, output = model.predict_instances(img, n_tiles=model._guess_n_tiles(img))
#         break
   
#     print('Model done predicting!')
    
#     print(len(output['points']), y_test[0])
#     print(output)
#     print(mask.shape)