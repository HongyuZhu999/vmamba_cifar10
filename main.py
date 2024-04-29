# Use Command in Terminals: python main.py --cfg configs\cifar10.yaml
import torch
import torchvision
import torchvision.transforms as transforms
from models import build_model
from config import get_config
import argparse
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import os

# Configs for Cifar10
EPOCHS = 100
TRAIN_LR = 0.01


# CIFAR-10 Data Load
def load_cifar10(batch_size, data_dir='data/cifar10'):
    # Normalize the CIFAR-10 dataset with the mean and standard deviation of CIFAR-10 images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Load the training, validation, and test sets
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(train_set, [48000, 2000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, huffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument(
        '--cfg',
        type=str, required=True,
        metavar="FILE",
        help='path to config file'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def setup_training(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR)
    return criterion, optimizer


def save_model(model, path):
    torch.save(model.state_dict(), path)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,
                folder_name, current_time):
    model.cuda()
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                running_loss = 0.0

        # Validate
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        print(f'Epoch {epoch + 1}/{num_epochs}: Loss: {running_loss / 200:.3f}, Validation Loss: {val_loss:.3f},'
              f' Validation Accuracy: {val_accuracy:.2f}%')

    # Save model
    model_file = os.path.join(folder_name, f'model_{current_time}.pth')
    save_model(model, model_file)
    print(f"Model saved in {model_file}")


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    if torch.cuda.is_available():
        print("GPU is available!")
    else:
        print("GPU is not available. Training will be done on CPU.")

    args, config = parse_option()
    model = build_model(config)
    # print(model)

    # CIFAR-10
    train_loader, val_loader, test_loader = load_cifar10(config.DATA.BATCH_SIZE, 'data/cifar10')

    # Create model save folder
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f'model_{current_time}'
    os.makedirs(folder_name, exist_ok=True)

    # Train
    print("Start Training")
    criterion, optimizer = setup_training(model)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS,
                folder_name=folder_name, current_time=current_time)

    # eval
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%')


if __name__ == "__main__":
    main()
