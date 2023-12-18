import argparse
import torch
import tkinter as tk
from tkinter import filedialog
import seaborn as sns
import matplotlib.pyplot as plt

def tensor_operations(tensor):
  
    tensor_sum = torch.sum(tensor)
    tensor_mean = torch.mean(tensor)

    return tensor_sum, tensor_mean

def visualize_tensor(tensor):
    # Flatten the tensor for visualization
    flat_tensor = tensor.view(-1).numpy()

   
    sns.histplot(flat_tensor, kde=True)
    plt.title("Distribution of Tensor Values")
    plt.xlabel("Tensor Values")
    plt.ylabel("Frequency")
    plt.show()

def user_inputs():
    
    root = tk.Tk()
    root.withdraw()

    
    file_path = filedialog.askopenfilename(title="Select a file (not used in this example)")

    return torch.randn(3, 3)  

def main():
  
    tensor = user_inputs()

s
    tensor_sum, tensor_mean = tensor_operations(tensor)


    print(f"Sum of tensor elements: {tensor_sum}")
    print(f"Mean of tensor elements: {tensor_mean}")

    
    visualize_tensor(tensor)

if __name__ == '__main__':
    main()
