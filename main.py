# Use Command in Terminals: python main.py --cfg configs\cifar10.yaml
import torch
import torchvision
import torchvision.transforms as transforms
from models import build_model
from config import get_config
import argparse
import torch.optim as optim
import torch.nn as nn


def load_cifar10(batch_size, data_dir='data/cifar10'):
    # Normalize the CIFAR-10 dataset with the mean and standard deviation of CIFAR-10 images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Load the training and test sets
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    return train_loader, test_loader


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def setup_training(model, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 示例：每30个epoch减少到原来的0.1倍
    return criterion, optimizer, scheduler


def save_model(model, path):
    torch.save(model.state_dict(), path)


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
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
                print(f'Epoch {epoch + 1}: Loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    save_model(model, f'model_save/model.pth')


def evaluate_model(model, data_loader, criterion):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 在评估过程中不进行梯度计算
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
    args, config = parse_option()
    model = build_model(config)
    print(model)
    model.cuda()

    # CIFAR-10
    train_loader, test_loader = load_cifar10(config.DATA.BATCH_SIZE, 'data/cifar10')

    # Train
    criterion, optimizer, scheduler = setup_training(model, config)
    train_model(model, train_loader, criterion, optimizer, num_epochs=5)

    # eval
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%')


if __name__ == "__main__":
    main()
