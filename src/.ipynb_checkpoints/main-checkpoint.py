import torch
import IPython
from notebook.app import LabServerApp

def gpu_check():
    print(f"GPU is avaliable: {torch.cuda.is_available()}")
    print(f"GPU device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"GPU device name: {torch.cuda.get_device_name(current_device)}")
        torch.set_default_device('cuda')
        print("Device set to GPU")
    else:
        print("Device default to CPU") 

def launch_jupyter():
   LabServerApp.launch_instance([])

if __name__ == '__main__':
    gpu_check()
    launch_jupyter()