import tensorflow as tf
import panel as pn
from app import app

def gpu_check():
    gpus = tf.config.list_physical_devices('GPU')
    print()
    if gpus:
        print("GPU is available")
        print(f"GPUs {gpus}")
    else:
        print("GPU unavailable")
    print()
    
def launch():
    pn.serve({"app.py":app}, port=5006)

if __name__ == '__main__':
    gpu_check()
    launch()