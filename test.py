import torch
import matplotlib.pyplot as plt
from model import CNNModel  # 导入你的模型
from data_loader import test_loader

# 加载模型
model = CNNModel()

# 记录测试准确率
test_accuracy = []

# 在测试集上评估模型
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracy.append(accuracy)
    print('Accuracy on the test images: %.2f %%' % accuracy)

# 可视化测试准确率
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.savefig('test_accuracy.png')  # 保存为文件
plt.close()
