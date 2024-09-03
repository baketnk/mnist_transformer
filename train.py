import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
writer = SummaryWriter('runs/mnist_experiment' + datetime.now().isoformat())
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def calculate_class_accuracy(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    return {i: acc for i, acc in enumerate(class_accuracies)}

def train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=1e-4, weight_decay=0.0001):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    class_count = model.class_count

    accumulation_steps = 1
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) # type: ignore

    class_sample_count = torch.tensor([(y_train == t).sum() for t in range(class_count)])
    weight = 1. / class_sample_count.float()
    class_weights = weight / weight.sum()
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = FocalLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    #
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    best_val_loss = float('inf')
    no_improve_epochs = 0
    for epoch in tqdm(range(epochs), unit="epoch"):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        optimizer.zero_grad()
        
        for i in tqdm(range(0, len(X_train), batch_size)):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss = loss / accumulation_steps
            if i == 0 and epoch <= 25:  # Only for the first batch
                
                assert (1.0 >= batch_X[0][0][1][0]) and (batch_X[0][0][1][0] >= 0.0)
                print()
                print(f"first prediction tensor: {outputs[0]}")
                print(f"Predicted classes: {outputs.argmax(dim=1)[:16]}")
                print(f"   Actual classes: {batch_y[:16]}")
                print(f"Loss: {loss.item()}")
            if torch.isnan(loss):
                print("NaN loss")
                break
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            outputs, _ = model(batch_X)
            _, predicted = outputs.max(1)

            total += batch_y.size(0)
            total_loss += loss.item() * accumulation_steps
            correct += predicted.eq(batch_y).sum().item()

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if total_norm > 1.0:
                print(f"Epoch {epoch+1}, Batch {i//batch_size}: Gradient norm: {total_norm:.4f} (clipped to 1.0)")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        if param_norm > 0.1:  # Adjust this threshold as needed
                            print(f"  Large gradient in {name}: {param_norm:.4f}")
       
        # Make sure to perform the final optimization step if needed
        if (len(X_train) / batch_size) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {batch_size*total_loss/len(X_train):.4f}, Train Accuracy: {train_accuracy:.2f}%')
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'gradients/{name}', param.grad, epoch) 



        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_true_labels = []
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i+batch_size].to(device)
                batch_y = y_val[i:i+batch_size].to(device)
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(batch_y.cpu().numpy())

        val_accuracy = 100. * val_correct / val_total
        val_loss =val_loss*batch_size/len(X_val)
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')



        val_f1_score = calculate_f1_score(val_true_labels, val_predictions)
        class_accuracies = calculate_class_accuracy(val_true_labels, val_predictions, class_count)

        print(f'Validation F1 Score: {val_f1_score:.4f}')
        print('Class-wise Accuracy:')
        for class_idx, accuracy in class_accuracies.items():
            print(f'  Class {class_idx}: {accuracy:.4f}')

# Log F1 score and class-wise accuracies
        writer.add_scalar('Metrics/val_f1_score', val_f1_score, epoch)
        for class_idx, accuracy in class_accuracies.items():
            writer.add_scalar(f'Metrics/class_{class_idx}_accuracy', accuracy, epoch)

        # Add this after calculating val_loss
        writer.add_histogram('val_predictions', np.array(val_predictions), epoch)
        writer.add_histogram('val_true_labels', np.array(val_true_labels), epoch)
        
        # Log model parameters and gradients
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    
        scheduler.step(val_loss)
        # scheduler.step()
        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint if it's the best so far
        if val_loss < best_val_loss:
            no_improve_epochs = 0
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model_checkpoint.pth')
            print(f"Saved new best model checkpoint with validation loss: {best_val_loss:.4f}")
        else:
            no_improve_epochs += 1

        # Early stopping
        if epoch >= 40 and no_improve_epochs >= 10:
            print("Early stopping due to low validation accuracy")
            break
