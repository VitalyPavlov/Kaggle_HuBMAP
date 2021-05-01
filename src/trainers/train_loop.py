

from torch.cuda.amp import autocast

def one_epoch(model, criterion, optimizer, 
               dataloaders, scaler, device):
        model.train()
        running_loss = 0.0
        train_size = 0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * inputs.size(0)
            train_size += inputs.size(0)

        epoch_loss_train = running_loss / train_size

        return epoch_loss_train