import torch
from sklearn.metrics import f1_score

def train(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs):
    print_every = 9999
    patience = 9999
    epochs_increasing = 0
    last_loss = 100000

    train_losses = []
    val_losses = []
    test_losses = []
    train_f1 = []
    test_f1 = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        y_train_true = []
        y_train_pred_proba = []

        for inputs, labels in train_loader:
            model.train()
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            y_train_true.extend(labels.cpu().tolist())
            y_train_pred_proba.extend(output.cpu().tolist())
            running_loss += loss.item()

        running_loss /= len(train_loader)
        y_train_true = torch.Tensor(y_train_true)
        y_train_pred_proba = torch.Tensor(y_train_pred_proba)
        train_losses.append(running_loss)
        train_pred = (torch.Tensor(y_train_pred_proba) > 0.5).float()
        train_f1.append(f1_score(y_train_true, train_pred))

        test_loss = 0.0
        y_test_true = []
        y_test_pred_proba = []
        for inputs, labels in test_loader:
            model.eval()
            output = model(inputs)
            loss = criterion(output, labels)
            test_loss += loss.item()
            y_test_true.extend(labels.cpu().tolist())
            y_test_pred_proba.extend(output.cpu().tolist())

        test_loss /= len(test_loader)
        y_test_true = torch.Tensor(y_test_true)
        y_test_pred_proba = torch.Tensor(y_test_pred_proba)
        test_losses.append(test_loss)
        test_pred = (torch.Tensor(y_test_pred_proba) > 0.5).float()
        test_f1.append(f1_score(y_test_true, test_pred))

        val_loss = 0.0
        for inputs, labels in val_loader:
            model.eval()
            output = model(inputs)
            loss = criterion(output, labels)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        if val_loss > last_loss:
            epochs_increasing += 1
            if epochs_increasing >= patience:
                print('Early stopping!\n')
                metrics = [train_losses, val_losses, test_losses, train_f1, test_f1]
                return y_test_true, test_pred, metrics
        else:
            epochs_increasing = 0

        last_loss = val_loss
        if epoch % print_every == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {running_loss:.4f}, Test loss: {test_loss:.4f}")
            
    metrics = [train_losses, val_losses, test_losses, train_f1, test_f1]

    # Return the collected metrics
    return y_test_true, test_pred, metrics