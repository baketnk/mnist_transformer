

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid




def visualize(model, X_test, y_test, num_samples=5):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Select a few random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    samples = X_test[indices].to(device)
    labels = y_test[indices]

    with torch.no_grad():
        # Patch embedding visualization
        patch_embed = model.patch_embed(samples)
        
        # Get attention maps (you might need to modify the model to return attention weights)
        logits, attention_maps = model(samples, return_attention=True)

        # Position embedding visualization
        pos_embed = model.pos_embed[0, 1:].reshape(-1, 7, 7)

    # Plot results
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(samples[i].cpu().squeeze(), cmap='gray')
        axes[i, 0].set_title(f"Original (Label: {labels[i]})")
        
        # Patch embedding
        axes[i, 1].imshow(make_grid(patch_embed[i].unsqueeze(1), nrow=7).cpu().permute(1, 2, 0))
        axes[i, 1].set_title("Patch Embedding")
        
        # Attention map (using the last layer, last head)
        axes[i, 2].imshow(attention_maps[-1][i, -1].cpu(), cmap='viridis')
        axes[i, 2].set_title("Attention Map")
        
        # Classification confidence
        probs = F.softmax(logits[i], dim=0)
        axes[i, 3].bar(range(10), probs.cpu())
        axes[i, 3].set_title("Class Probabilities")
        axes[i, 3].set_xticks(range(10))

    plt.tight_layout()
    plt.savefig('visualization.png')
    print("Visualization saved as 'visualization.png'")

    # Visualize position embeddings
    plt.figure(figsize=(10, 10))
    plt.imshow(make_grid(pos_embed.unsqueeze(1), nrow=7).cpu().permute(1, 2, 0))
    plt.title("Position Embeddings")
    plt.savefig('position_embeddings.png')
    print("Position embeddings saved as 'position_embeddings.png'")
