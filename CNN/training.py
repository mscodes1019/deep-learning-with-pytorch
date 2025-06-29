import torch
from tqdm.notebook import tqdm


def train_epoch(model, optimizer, loss_fn, data_loader, device="cpu"):
    # INSERT ...
    # REMOVE{
    training_loss = 0.0
    model.train()

    # Iterate over all batches in the training set to complete one epoch
    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)

        output = model(inputs)
        loss = loss_fn(output, targets)

        loss.backward()
        optimizer.step()
        training_loss += loss.data.item() * inputs.size(0)

    return training_loss / len(data_loader.dataset)
    # REMOVE}


def predict(model, data_loader, device="cpu"):
    # INSERT ...
    # REMOVE{
    all_probs = torch.tensor([]).to(device)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs.to(device)
            output = model(inputs)
            probs = torch.nn.functional.softmax(output, dim=1)
            all_probs = torch.cat((all_probs, probs), dim=0)

    return all_probs
    # REMOVE}


def score(model, data_loader, loss_fn, device="cpu"):
    # INSERT ...
    # REMOVE{
    total_loss = 0
    total_correct = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Scoring", leave=False):
            inputs = inputs.to(device)
            output = model(inputs)

            targets = targets.to(device)
            loss = loss_fn(output, targets)
            total_loss += loss.data.item() * inputs.size(0)

            correct = torch.eq(torch.argmax(output, dim=1), targets)
            total_correct += torch.sum(correct).item()
    average_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / len(data_loader.dataset)
    return average_loss, accuracy
    # REMOVE}


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    training_losses = []
    val_losses = []
    epoch_nums = []

    for epoch in range(1, epochs + 1):
        training_loss = train_epoch(model, optimizer, loss_fn, train_loader, device)
        val_loss, val_accuracy = score(model, val_loader, loss_fn, device)

        training_losses.append(training_loss)
        val_losses.append(val_loss)
        epoch_nums.append(epoch)

        print(
            f"Epoch: {epoch}, Training Loss: {training_loss:.2f}, "
            f"Test Loss: {val_loss:.2f}, Test accuracy = {val_accuracy:.2f}"
        )

    return epoch_nums, training_losses, val_losses
