import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import KFold
from models.classifier import FCDDetector
from data.bonn_data import BonnMRIClassificationDataset

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='training_log.txt')
logger = logging.getLogger(__name__)

def train_and_validate_fold(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Train and validate a single fold of the cross-validation.
    
    Returns:
    - Dictionary of performance metrics for the fold
    """
    train_losses = []
    val_losses = []
    precisions = []
    recalls = []
    val_accuracies = []
    f1_scores = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader, 1):
            # Clear cache before each batch
            torch.cuda.empty_cache()
            
            inputs = batch['FLAIR'].float().to(device)
            labels = batch['label'].float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print training progress
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # Free up memory
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Initialize metric accumulators for entire validation set
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0

        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(val_loader, 1):
                # Clear cache before each batch
                torch.cuda.empty_cache()
                
                val_inputs = val_batch['FLAIR'].float().to(device)
                val_labels = val_batch['label'].float().to(device)

                val_outputs = model(val_inputs).squeeze(1)
                val_loss += criterion(val_outputs, val_labels).item()

                # Calculate metrics
                predicted = (torch.sigmoid(val_outputs) > 0.5).float()
                correct_predictions += (predicted == val_labels).sum().item()
                total_predictions += val_labels.size(0)

                # Accumulate confusion matrix components across all validation batches
                batch_true_positives = ((predicted == 1) & (val_labels == 1)).sum().item()
                batch_false_positives = ((predicted == 1) & (val_labels == 0)).sum().item()
                batch_false_negatives = ((predicted == 0) & (val_labels == 1)).sum().item()
                
                total_true_positives += batch_true_positives
                total_false_positives += batch_false_positives
                total_false_negatives += batch_false_negatives

                # Calculate batch-level metrics for progress reporting only
                batch_precision = batch_true_positives / (batch_true_positives + batch_false_positives) if (batch_true_positives + batch_false_positives) > 0 else 0
                batch_recall = batch_true_positives / (batch_true_positives + batch_false_negatives) if (batch_true_positives + batch_false_negatives) > 0 else 0
                batch_f1 = 2 * (batch_precision * batch_recall) / (batch_precision + batch_recall) if (batch_precision + batch_recall) > 0 else 0

                # Print validation progress (batch-level metrics for monitoring)
                print(f"Validation Batch [{val_batch_idx}/{len(val_loader)}], "
                      f"Loss: {val_loss/(val_batch_idx):.4f}, Batch Precision: {batch_precision:.4f}, Batch Recall: {batch_recall:.4f}, Batch F1: {batch_f1:.4f}")

                # Free up memory
                del val_inputs, val_labels, val_outputs
                torch.cuda.empty_cache()

            val_accuracy = correct_predictions / total_predictions
            
            # Calculate final metrics across entire validation set
            precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
            recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Store metrics
            train_losses.append(running_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            precisions.append(precision)
            recalls.append(recall)
            val_accuracies.append(val_accuracy)
            f1_scores.append(f1)

            # Log progress
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Train Loss: {running_loss / len(train_loader):.4f}, "
                        f"Validation Loss: {val_loss / len(val_loader):.4f}, "
                        f"Validation Accuracy: {val_accuracy:.4f}, "
                        f"Precision: {precision:.4f}, "
                        f"Recall: {recall:.4f}, "
                        f"F1 Score: {f1:.4f}")

    return {
        'train_losses': torch.tensor(train_losses),
        'val_losses': torch.tensor(val_losses),
        'precisions': torch.tensor(precisions),
        'recalls': torch.tensor(recalls),
        'val_accuracies': torch.tensor(val_accuracies),
        'f1_scores': torch.tensor(f1_scores)
    }

def k_fold_cross_validation(model, data, k_folds=5, num_epochs=30, learning_rate=0.001, batch_size=4):
    """
    Perform K-Fold Cross Validation with memory-efficient approaches
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    print(f"Using device: {device}")
    # Prepare for K-Fold Cross-Validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print("K-Fold initialized")

    # Dictionary to store results for each fold
    fold_results = {}
    
    # Prepare optimizer and loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Iterate through folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(data), 1):
        logger.info(f"FOLD {fold}")
        
        print(f"FOLD {fold}")

        # Reset model weights for each fold
        model.reset_parameters()
        
        # Move model to device
        model = model.to(device)
        
        # Create samplers for this fold
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        # Create data loaders for this fold with reduced batch size
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
        val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sampler, num_workers=0, pin_memory=True)
        
        # Reinitialize optimizer for each fold with trainable parameters
        optimizer = optim.Adam(model.get_trainable_parameters(), lr=learning_rate)
        
        # Train and validate this fold
        fold_metrics = train_and_validate_fold(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
        
        # Save model weights for this fold
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': fold_metrics
        }, f'model_fold_{fold}_checkpoint.pth')
        
        # Store fold results
        fold_results[f'fold_{fold}'] = fold_metrics
        
        # Log average performance for this fold
        logger.info(f"Fold {fold} Average Metrics:")
        logger.info(f"Validation Accuracy: {fold_metrics['val_accuracies'].mean():.4f}")
        logger.info(f"Precision: {fold_metrics['precisions'].mean():.4f}")
        logger.info(f"Recall: {fold_metrics['recalls'].mean():.4f}")
        logger.info(f"F1 Score: {fold_metrics['f1_scores'].mean():.4f}")
        
        # Clear CUDA cache after each fold
        torch.cuda.empty_cache()
    
    # Save complete results
    torch.save(fold_results, 'k_fold_cross_validation_results.pth')
    
    return fold_results

# Usage example
def main():
    # Initialize model
    model = FCDDetector()
    
    # Load dataset
    data = BonnMRIClassificationDataset('./Dataset')

    # Perform cross-validation
    k_fold_cross_validation(
        model, 
        data, 
        k_folds=5,  # Increase number of folds
        num_epochs=30,  # Default number of epochs
        learning_rate=0.001,
        batch_size=16  # Reduced batch size to handle memory constraints
    )

if __name__ == '__main__':
    main()