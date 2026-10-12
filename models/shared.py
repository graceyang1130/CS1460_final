import torch
from tqdm import tqdm

def train_model(model, train_dataloader, val_dataloader, device='cuda', epochs=1, checkpoint='climate.pt'):
    model.train()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_losses = []
    val_losses = []
    min_val_loss = None

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_dataloader)
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description(f"train loss: {running_loss/(i+1)}")
        train_loss = running_loss / len(train_dataloader)

        running_loss = 0.0
        total = 0
        correct = 0
        pbar = tqdm(val_dataloader)
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            pbar.set_description(f"val loss: {running_loss/(i+1)}, acc: {correct/total}")
        val_loss = running_loss / len(val_dataloader)
        acc = correct / total
        if min_val_loss is None or val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'./checkpoints/{checkpoint}')

        print(f"Epoch {epoch+1}: train loss={train_loss}, val_loss={val_loss}, acc={acc}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return model, train_losses, val_losses
