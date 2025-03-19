import torch
import torch.nn as nn
import torch.optim as optim

# Heavy Ball Method with manual backpropagation and adaptive learning rate
def heavy_ball_manual(model, dataloader, max_iter=100, alpha=0.01, beta=0.9, l1_lambda=0.01):

    prev_update = [torch.zeros_like(param) for param in [model.w1, model.b1, model.w2, model.b2]]
    history = []  # Loss history
    prev_loss = float('inf')
    
    for epoch in range(max_iter):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            # Forward
            outputs = model.forward(inputs)

            # MSE loss
            mse_loss = ((outputs - targets) ** 2).mean()
            l1_loss = l1_lambda * (torch.abs(model.w1).sum() + torch.abs(model.w2).sum())
            total_loss = mse_loss + l1_loss
            epoch_loss = total_loss.item()

            # backward pass
            model.backward(inputs, targets, outputs, l1_lambda=l1_lambda)

            # Adaptive learning rate
            if epoch >= 5 and epoch_loss > prev_loss:
                alpha *= 0.9999
            
            # Update parameters
            model.update_params(alpha, beta, prev_update)
        
        # Record loss for the epoch
        history.append(epoch_loss / len(dataloader))
        prev_loss = epoch_loss

        print(f'Epoch {epoch+1}/{max_iter}, Loss: {history[-1]:.4f}, Alpha: {alpha:.6f}')
    
    return model, history


# 2 hidden layers version
def heavy_ball_manual_2l(model, dataloader, max_iter=100, alpha=0.01, beta=0.9, l1_lambda=0.01):

    prev_update = [torch.zeros_like(param) for param in [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3]]
    history = []  # Loss history
    prev_loss = float('inf')
    
    for epoch in range(max_iter):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            # Forward
            outputs = model.forward(inputs)

            # MSE loss
            mse_loss = ((outputs - targets) ** 2).mean()
            l1_loss = l1_lambda * (torch.abs(model.w1).sum() + torch.abs(model.w2).sum() + torch.abs(model.w3).sum())
            total_loss = mse_loss + l1_loss
            epoch_loss = total_loss.item()

            #backward pass
            model.backward(inputs, targets, outputs, l1_lambda=l1_lambda)

            # Adaptive learning rate
            if epoch >= 5 and epoch_loss > prev_loss:
                alpha *= 0.9999
            
            # Update parameters
            model.update_params(alpha, beta, prev_update)
        
        # Record loss for the epoch
        history.append(epoch_loss / len(dataloader))
        prev_loss = epoch_loss

        print(f'Epoch {epoch+1}/{max_iter}, Loss: {history[-1]:.4f}, Alpha: {alpha:.6f}')
    
    return model, history
