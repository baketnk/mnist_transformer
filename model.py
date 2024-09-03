import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from infer import inference
from train import train
from visualize import visualize
import math

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.class_count = 10
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, None

class FeedForwardLayer(nn.Module):
    def __init__(
            self, 
            n_embed: int, 
            extend_width: int=4, 
            dropout: float=0.1
        ):
        super(FeedForwardLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_embed, extend_width*n_embed), 
            nn.ReLU(),
            nn.Linear(extend_width*n_embed, n_embed), 
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.view(-1, x.size(-1))
        x = self.layer(x)
        return x.view(original_shape)



class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.head_dim = n_embed // num_heads
        
        self.q_linear = nn.Linear(n_embed, n_embed)
        self.k_linear = nn.Linear(n_embed, n_embed)
        self.v_linear = nn.Linear(n_embed, n_embed)
        self.out_linear = nn.Linear(n_embed, n_embed)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_embed)
        output = self.out_linear(context)
        
        return output, attn_weights

class TBlock(nn.Module):
    def __init__(self, n_embed: int, num_heads: int, dropout = 0.2, bias = True):
        super(TBlock, self).__init__()
        self.mhsa = MultiHeadAttention(n_embed, num_heads, dropout)
        self.feed = FeedForwardLayer(n_embed, dropout=dropout)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        attn_out, attn_weights = self.mhsa(x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.feed(x))
        if return_attention:
            return x, attn_weights
        return x, None

class MNISTDetector(nn.Module):
    def __init__(self, input_size: int, n_embed: int, num_heads: int, n_layers: int, class_count: int, patch_size: int = 7, patch_stride: int = 7, dropout: float = 0.1):
        super(MNISTDetector, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.class_count = class_count
        self.patch_stride = patch_stride
        self.dropout = dropout
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, n_embed, kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm2d(n_embed),
            nn.GELU()
        )


        # Position embedding
        num_patches = (((input_size - patch_size) // patch_stride) + 1) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches+1, n_embed) * 0.02, requires_grad=True)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embed), requires_grad=True)
        self.embed_drop = nn.Dropout(dropout)
        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TBlock(num_heads=num_heads, n_embed=n_embed, dropout=dropout) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(n_embed)
        self.classifier = nn.Linear(n_embed, class_count)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, mean=0.0, std=0.02)
            nn.init.trunc_normal_(m, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embedding
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.embed_drop(x)
        
        # Apply transformer blocks
        attention_maps = []
        for block in self.blocks:
            x, attn = block(x, return_attention=return_attention)
            if return_attention:
                attention_maps.append(attn)
        
        # Normalize and classify
        x = self.norm(x)
        x = x[:, 0]  # Use only the class token
        logits = self.classifier(x)
        
        if return_attention:
            return logits, attention_maps
        return logits, None

def shuffle_data(X, y):
    indices = torch.randperm(X.shape[0])
    return X[indices], y[indices]


def analyze_data(X_train, y_train, X_test, y_test):
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    print(f"\nX_train dtype: {X_train.dtype}")
    print(f"y_train dtype: {y_train.dtype}")
    
    print(f"\nX_train min: {X_train.min()}, max: {X_train.max()}")
    print(f"y_train min: {y_train.min()}, max: {y_train.max()}")
    
    print("\nLabel distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {label}: {count} {100.0*count/len(y_train):.2f}")
    
    print("\nFirst few labels:", y_train[:20])

    print("\nUnique labels in training set:", np.unique(y_train))
    print("Unique labels in test set:", np.unique(y_test))

    # Visualize a few images
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y_train[i]}")
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("Sample images saved as 'sample_images.png'")


def load_mnist_data(file_path):
    mat = scipy.io.loadmat(file_path)
    data = mat['data'].T  # Transpose to get (samples, features)
    labels = mat['label'].squeeze()  # Remove unnecessary dimensions
    # data, labels = shuffle_data(data, labels)
   
    print("Raw labels shape:", labels.shape)
    print("First few raw labels:", labels[:20])

    # Split into train and test sets (assuming 60,000 training samples as per standard MNIST)
    # train_len = int(len(labels) * 0.9)
    train_len = 60000
    val_len = len(labels) - train_len
    X_train = data[:train_len]
    y_train = labels[:train_len]
    X_test = data[train_len:]
    y_test = labels[train_len:]
    assert len(y_test) == val_len
    return X_train, y_train, X_test, y_test

def preprocess_data(X, y):
    X = torch.from_numpy(X).float().view(-1, 1, 28, 28) / 255.0 # Reshape to (B, C, H, W)

    X, y = shuffle_data(X, y)
    analyze_data(X_train, y_train, X_test, y_test) 
    y = torch.from_numpy(y).long()
    return X, y 


def load_checkpoint(model, optimizer=None, filename='best_model_checkpoint.pth'):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=[
            "train",
            "run",
            "visualize"
        ]
    )


    args = parser.parse_args()
    
    input_size = 28
    n_embed = 128
    num_heads = 8
    block_size = n_embed // num_heads
    n_layers = 6
    class_count = 10
    dropout = 0.2

    lr = 1e-3
    wd = 0.01
    epoch_limit = 100
    model = MNISTDetector(
            input_size=input_size,
            n_embed=n_embed,
            num_heads=num_heads,
            n_layers=n_layers,
        class_count=class_count,
        patch_size=4,
        patch_stride=4,
        dropout=dropout,
        )
    model = model.to("mps")
    # think this is wrong
    # inp = torch.ones((28,28))
    inp = torch.ones((1, 1, 28, 28), dtype=torch.float).to("mps")
    out, _ = model(inp)
    # print(out.shape)

    if args.command == "run":
        model, _, _, _ = load_checkpoint(model)
         # Load and preprocess your test data
        _, _, X_test, y_test = load_mnist_data('data/mnist-original.mat')
        X_test, y_test = preprocess_data(X_test, y_test)
        
        # Run inference
        predictions, expected = inference(model, X_test, y_test)
         
        # Calculate and print accuracy
        accuracy = (predictions == y_test.numpy()).mean()
        print(f"Test Accuracy: {accuracy:.4f}")

        print("\nFirst 20 comparisons (Expected > Calculated):")
        for i in range(20):
            print(f"{expected[i]} > {predictions[i]}")

        # Print out misclassifications
        misclassified = np.where(predictions != expected)[0]
        print("\nMisclassified examples (Expected > Calculated):")
        for i in misclassified[:20]:  # Print first 20 misclassifications
            print(f"{expected[i]} > {predictions[i]}")


    elif args.command == "train":
        X_train, y_train, X_test, y_test = load_mnist_data('data/mnist-original.mat')
        X_val, y_val = preprocess_data(X_test, y_test)
        X_train, y_train = preprocess_data(X_train, y_train)
        # model = SimpleCNN()
        train(model, X_train, y_train, X_val, y_val, 
              epochs=epoch_limit, 
              batch_size=256, 
              lr=lr,
              weight_decay=wd)
    elif args.command == "visualize":
        model, _, _, _ = load_checkpoint(model)
        _, _, X_test, y_test = load_mnist_data('data/mnist-original.mat')
        X_test, y_test = preprocess_data(X_test, y_test)
        visualize(model, X_test, y_test)
    else:
        parser.print_help()
        raise ValueError("bad command value")






