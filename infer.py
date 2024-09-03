

import torch
import numpy as np

def inference(model, X_test, y_test, batch_size=32):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    predictions = []
    expected = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size]
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
            expected.extend(batch_y.numpy())
    
    return np.array(predictions), np.array(expected)
