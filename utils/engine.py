import torch
import torch.nn.functional as F

def calculate_topk_accuracy(y_pred: torch.Tensor, y: torch.Tensor, k: int = 5) -> tuple:
    """Calculate top-k accuracy.
    
    Args:
        y_pred (torch.Tensor): predicted labels.
        y (torch.Tensor): true labels.
        k (int): the number of top predictions to consider.

    Returns:
        tuple: tuple containing:
            acc_1 (torch.Tensor): accuracy of top-1 prediction.
            acc_k (torch.Tensor): accuracy of top-k prediction.
    """
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

def train(model: torch.nn.Module, iterator: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module, scheduler: torch.optim.lr_scheduler._LRScheduler, device: str) -> tuple:
    """Train the model.
    
    Args:
        model (torch.nn.Module): the model to be trained.
        iterator (torch.utils.data.DataLoader): iterator over the training data.
        optimizer (torch.optim.Optimizer): optimizer used for training.
        criterion (torch.nn.Module): loss function used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler.
        device (str): device used for computation.

    Returns:
        tuple: tuple containing:
            epoch_loss (float): average loss per epoch.
            epoch_acc_1 (float): average top-1 accuracy per epoch.
            epoch_acc_5 (float): average top-5 accuracy per epoch.
    """
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        
        scheduler.step()
        
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def evaluate(model, iterator, criterion, device):
    """Evaluate the performance of a model on a validation or test set.
    
    Args:
        model (torch.nn.Module): The trained model to evaluate.
        iterator (torch.utils.data.DataLoader): The data loader for the validation or test set.
        criterion (torch.nn.modules.loss._Loss): The loss function to use for the evaluation.
        device (torch.device): The device on which to perform the evaluation.

    Returns:
        A tuple containing the average loss and top-1 and top-5 accuracies for the evaluation set.
    """

    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def epoch_time(start_time, end_time):
    """Compute the elapsed time between two time points.
    
    Args:
        start_time (float): The start time in seconds.
        end_time (float): The end time in seconds.

    Returns:
        A tuple containing the elapsed minutes and seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_predictions(model, iterator, device):
    """
    Get predictions from a PyTorch model.

    Args:
        model (torch.nn.Module): PyTorch model.
        iterator (torch.utils.data.DataLoader): PyTorch data loader.

    Returns:
        tuple: Tuple containing images, labels, and probabilities.
    """

    # Set model to evaluation mode
    model.eval()

    # Create empty lists for images, labels, and probabilities
    images = []
    labels = []
    probs = []

    # Turn off gradient calculations for efficiency
    with torch.no_grad():

        # Iterate through data loader
        for (x, y) in iterator:

            # Move inputs to GPU
            x = x.to(device)

            # Get predictions from model
            y_pred = model(x)

            # Apply softmax function to predictions
            y_prob = F.softmax(y_pred, dim=-1)

            # Get the predicted label with the highest probability
            top_pred = y_prob.argmax(1, keepdim=True)

            # Append inputs, labels, and probabilities to lists
            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    # Concatenate inputs, labels, and probabilities along the batch dimension
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    # Return images, labels, and probabilities
    return images, labels, probs