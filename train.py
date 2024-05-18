import torch
import torch.optim as optim
import torch.nn as nn
from model import SimpleCNN
from load_data import get_dataloader
import matplotlib.pyplot as plt


def train_model(model, train_loader, criterion, optimizer, num_epochs=25, device='cpu'):
    model.to(device)
    model.train()
    train_loss = []
    train_accuracy = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # 调整输出和标签的形状以匹配
            outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, model.num_classes)
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            # 计算训练正确率
            _, preds = torch.max(outputs, 1)
            correct_pixels += (preds == labels).sum().item()
            total_pixels += labels.numel()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_pixels / total_pixels
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    # 绘制训练损失和正确率曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracy, label='Train Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_loss_accuracy_curve.png')
    plt.show()


if __name__ == "__main__":
    train_loader = get_dataloader('raw_data', 'groundtruth', batch_size=16)
    model = SimpleCNN(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, criterion, optimizer, num_epochs=25, device=device)
    torch.save(model.state_dict(), 'model.pth')




# import torch
# import torch.optim as optim
# import torch.nn as nn
# from model import SimpleCNN
# from load_data import get_dataloader
# import matplotlib.pyplot as plt
#
#
# def train_model(model, train_loader, criterion, optimizer, num_epochs=25, device='cpu'):
#     model.to(device)
#     model.train()
#     train_loss = []
#     train_accuracy = []
#
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct_pixels = 0
#         total_pixels = 0
#
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(images)
#
#             # 调整输出和标签的形状以匹配
#             outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, model.num_classes)
#             labels = labels.view(-1)
#
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * images.size(0)
#
#             # 计算训练正确率
#             _, preds = torch.max(outputs, 1)
#             correct_pixels += (preds == labels).sum().item()
#             total_pixels += labels.numel()
#
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_accuracy = correct_pixels / total_pixels
#         train_loss.append(epoch_loss)
#         train_accuracy.append(epoch_accuracy)
#
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
#
#     # 绘制训练损失和正确率曲线
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Curve')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(range(1, num_epochs + 1), train_accuracy, label='Train Accuracy', color='orange')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Training Accuracy Curve')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.savefig('training_loss_accuracy_curve.png')
#     plt.show()
#
#
# if __name__ == "__main__":
#     train_loader = get_dataloader('raw_data', 'groundtruth', batch_size=16)
#     model = SimpleCNN(num_classes=3)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     train_model(model, train_loader, criterion, optimizer, num_epochs=25, device=device)
#     torch.save(model.state_dict(), 'model.pth')
#
# """
# D:\ProgramData\anaconda3\envs\py311\python.exe G:\我的云端硬盘\01研究生理论基础\计算机视觉：原理与应用\第三次作业\train.py
# Epoch 1/25, Loss: 0.8064, Accuracy: 0.7291
# Epoch 2/25, Loss: 0.5738, Accuracy: 0.8466
# Epoch 3/25, Loss: 0.4652, Accuracy: 0.8466
# Epoch 4/25, Loss: 0.3530, Accuracy: 0.8765
# Epoch 5/25, Loss: 0.3149, Accuracy: 0.9004
# Epoch 6/25, Loss: 0.2161, Accuracy: 0.9178
# Epoch 7/25, Loss: 0.2009, Accuracy: 0.9273
# Epoch 8/25, Loss: 0.1725, Accuracy: 0.9337
# Epoch 9/25, Loss: 0.1603, Accuracy: 0.9397
# Epoch 10/25, Loss: 0.1498, Accuracy: 0.9433
# Epoch 11/25, Loss: 0.2039, Accuracy: 0.9322
# Epoch 12/25, Loss: 0.1699, Accuracy: 0.9389
# Epoch 13/25, Loss: 0.1483, Accuracy: 0.9451
# Epoch 14/25, Loss: 0.1391, Accuracy: 0.9467
# Epoch 15/25, Loss: 0.1477, Accuracy: 0.9445
# Epoch 16/25, Loss: 0.1615, Accuracy: 0.9426
# Epoch 17/25, Loss: 0.1428, Accuracy: 0.9467
# Epoch 18/25, Loss: 0.1362, Accuracy: 0.9483
# Epoch 19/25, Loss: 0.1340, Accuracy: 0.9489
# Epoch 20/25, Loss: 0.1311, Accuracy: 0.9498
# Epoch 21/25, Loss: 0.1272, Accuracy: 0.9511
# Epoch 22/25, Loss: 0.1256, Accuracy: 0.9517
# Epoch 23/25, Loss: 0.1209, Accuracy: 0.9538
# Epoch 24/25, Loss: 0.1195, Accuracy: 0.9541
# Epoch 25/25, Loss: 0.1660, Accuracy: 0.9430
#
# Process finished with exit code 0
# """
#
#
#
# """
# D:\ProgramData\anaconda3\envs\py311\python.exe G:\我的云端硬盘\01研究生理论基础\计算机视觉：原理与应用\第三次作业\train.py
# Epoch 1/25, Loss: 0.7708
# Epoch 2/25, Loss: 0.5773
# Epoch 3/25, Loss: 0.5136
# Epoch 4/25, Loss: 0.4458
# Epoch 5/25, Loss: 0.4139
# Epoch 6/25, Loss: 0.4041
# Epoch 7/25, Loss: 0.2690
# Epoch 8/25, Loss: 0.1991
# Epoch 9/25, Loss: 0.1779
# Epoch 10/25, Loss: 0.2616
# Epoch 11/25, Loss: 0.2444
# Epoch 12/25, Loss: 0.1740
# Epoch 13/25, Loss: 0.1685
# Epoch 14/25, Loss: 0.1590
# Epoch 15/25, Loss: 0.1495
# Epoch 16/25, Loss: 0.1566
# Epoch 17/25, Loss: 0.1523
# Epoch 18/25, Loss: 0.1507
# Epoch 19/25, Loss: 0.1491
# Epoch 20/25, Loss: 0.1474
# Epoch 21/25, Loss: 0.1401
# Epoch 22/25, Loss: 0.1419
# Epoch 23/25, Loss: 0.1364
# Epoch 24/25, Loss: 0.1345
# Epoch 25/25, Loss: 0.1343
#
# Process finished with exit code 0
# """



# import torch
# import torch.optim as optim
# import torch.nn as nn
# from model import SimpleCNN
# from load_data import get_dataloader
#
#
# def train_model(model, train_loader, criterion, optimizer, num_epochs=25, device='cpu'):
#     model.to(device)
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#
#             # 检查标签值范围
#             if labels.min() < 0 or labels.max() >= model.num_classes:
#                 print(f"标签值超出范围: {labels.min()} ~ {labels.max()}")
#                 print(f"异常标签图像: {images}")
#                 print(f"异常标签: {labels}")
#                 continue
#
#             optimizer.zero_grad()
#             outputs = model(images)
#
#             # 调整输出和标签的形状以匹配
#             outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, model.num_classes)
#             labels = labels.view(-1)
#
#             # 确保输出和标签的形状一致
#             if outputs.size(0) != labels.size(0):
#                 print(f"Shape mismatch: outputs size = {outputs.size()}, labels size = {labels.size()}")
#                 continue
#
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * images.size(0)
#
#         epoch_loss = running_loss / len(train_loader.dataset)
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
#
#
# if __name__ == "__main__":
#     train_loader = get_dataloader('raw_data', 'groundtruth', batch_size=16)
#     model = SimpleCNN(num_classes=3)  # 根据你的数据集中的实际类别数设置 num_classes
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     train_model(model, train_loader, criterion, optimizer, num_epochs=25, device=device)
#     torch.save(model.state_dict(), 'model.pth')
#
# """
# lr =  0.0005
# Epoch 1/25, Loss: 0.8541
# Epoch 2/25, Loss: 0.6125
# Epoch 3/25, Loss: 0.5114
# Epoch 4/25, Loss: 0.4162
# Epoch 5/25, Loss: 0.3279
# Epoch 6/25, Loss: 0.2662
# Epoch 7/25, Loss: 0.2226
# Epoch 8/25, Loss: 0.1990
# Epoch 9/25, Loss: 0.1818
# Epoch 10/25, Loss: 0.1747
# Epoch 11/25, Loss: 0.1618
# Epoch 12/25, Loss: 0.1536
# Epoch 13/25, Loss: 0.1495
# Epoch 14/25, Loss: 0.1369
# Epoch 15/25, Loss: 0.1355
# Epoch 16/25, Loss: 0.1564
# Epoch 17/25, Loss: 0.1327
# Epoch 18/25, Loss: 0.1366
# Epoch 19/25, Loss: 0.1264
# Epoch 20/25, Loss: 0.1329
# Epoch 21/25, Loss: 0.1242
# Epoch 22/25, Loss: 0.1480
# Epoch 23/25, Loss: 0.1385
# Epoch 24/25, Loss: 0.1308
# Epoch 25/25, Loss: 0.1276
#
# Process finished with exit code 0
#
# """