import torch
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
import numpy as np
from model import FCN

def apply_masks_to_image(image_path, model, device):
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    transform = ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 获取模型预测
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        outputs = torch.sigmoid(outputs)  # 使用sigmoid激活函数

    # 假设输出有两个通道，分别代表煤和矸石
    coal_mask = outputs[0, 0] > 0.5
    waste_mask = outputs[0, 1] > 0.5

    # 将遮罩转换为PIL图像以便绘制
    coal_mask = coal_mask.cpu().numpy()
    waste_mask = waste_mask.cpu().numpy()

    # 创建绘制对象
    draw = ImageDraw.Draw(image)

    # 遍历像素，应用标记
    for y in range(coal_mask.shape[0]):
        for x in range(coal_mask.shape[1]):
            if coal_mask[y, x]:
                draw.point((x, y), fill=(255, 0, 0))  # 使用红色表示煤
            if waste_mask[y, x]:
                draw.point((x, y), fill=(0, 255, 0))  # 使用绿色表示矸石

    # 显示或保存图像
    image.show()

# 示例用法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCN(input_channels=3).to(device)
model.load_state_dict(torch.load("fcn_model_weights.pt"))
apply_masks_to_image("./raw_data/201.jpg", model, device)
