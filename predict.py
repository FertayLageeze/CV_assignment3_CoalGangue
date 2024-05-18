import torch
from torchvision import transforms
from load_data import get_dataloader
from model import SimpleCNN
from utils import predict_and_visualize

# 加载预训练模型
model = SimpleCNN(num_classes=3)
model.load_state_dict(torch.load('model0001.pth'))

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义数据转换
data_transforms = transforms.Compose([
    transforms.ToTensor()
])

# 加载数据
test_loader = get_dataloader('raw_data', 'groundtruth', batch_size=1, shuffle=False, transform=data_transforms, start_idx=200, end_idx=236)

# 进行预测并可视化
predict_and_visualize(model, test_loader, device=device)
