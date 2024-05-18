import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, start_idx=0, end_idx=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
        self.label_files = sorted(os.listdir(label_dir), key=lambda x: int(os.path.splitext(x)[0]))
        self.image_files = self.image_files[start_idx:end_idx]
        self.label_files = self.label_files[start_idx:end_idx]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # 输出加载的信息
        print(f"Loading image: {image_path}")
        print(f"Loading label: {label_path}")

        # 读取原图 (黑白图像)
        image = Image.open(image_path).convert('RGB')  # 将黑白图像转换为 RGB 模式

        # 读取标记图像
        label = Image.open(label_path).convert('P')  # 保持调色板模式

        image = image.resize((852, 480))
        label = label.resize((852, 480))

        if self.transform:
            image = self.transform(image)
            label = torch.tensor(np.array(label), dtype=torch.long)

        return image, label

def get_dataloader(image_dir, label_dir, batch_size=16, shuffle=True, transform=None, start_idx=0, end_idx=None):
    dataset = CustomDataset(image_dir, label_dir, transform=transform, start_idx=start_idx, end_idx=end_idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import numpy as np
#
# class CustomDataset(Dataset):
#     def __init__(self, image_dir, label_dir, transform=None, start_idx=0, end_idx=None):
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.image_files = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
#         self.label_files = sorted(os.listdir(label_dir), key=lambda x: int(os.path.splitext(x)[0]))
#         self.image_files = self.image_files[start_idx:end_idx]
#         self.label_files = self.label_files[start_idx:end_idx]
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.image_files[idx])
#         label_path = os.path.join(self.label_dir, self.label_files[idx])
#
#         # 输出加载的信息
#         print(f"Loading image: {image_path}")
#         print(f"Loading label: {label_path}")
#
#         # 读取原图 (黑白图像)
#         image = Image.open(image_path).convert('RGB')  # 将黑白图像转换为 RGB 模式
#
#         # 读取标记图像
#         label = Image.open(label_path).convert('P')  # 保持调色板模式
#
#         image = image.resize((852, 480))
#         label = label.resize((852, 480))
#
#         if self.transform:
#             image = self.transform(image)
#             label = torch.tensor(np.array(label), dtype=torch.long)
#
#         return image, label
#
# def get_dataloader(image_dir, label_dir, batch_size=16, shuffle=True, transform=None, start_idx=0, end_idx=None):
#     dataset = CustomDataset(image_dir, label_dir, transform=transform, start_idx=start_idx, end_idx=end_idx)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import numpy as np
#
# class CustomDataset(Dataset):
#     def __init__(self, image_dir, label_dir, transform=None, start_idx=0, end_idx=None):
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.image_files = sorted(os.listdir(image_dir))[start_idx:end_idx]
#         self.label_files = sorted(os.listdir(label_dir))[start_idx:end_idx]
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.image_files[idx])
#         label_path = os.path.join(self.label_dir, self.label_files[idx])
#
#         # 读取原图 (黑白图像)
#         image = Image.open(image_path).convert('RGB')  # 将黑白图像转换为 RGB 模式
#
#         # 读取标记图像
#         label = Image.open(label_path).convert('P')  # 保持调色板模式
#
#         image = image.resize((852, 480))
#         label = label.resize((852, 480))
#
#         if self.transform:
#             image = self.transform(image)
#             label = torch.tensor(np.array(label), dtype=torch.long)
#
#         return image, label
#
# def get_dataloader(image_dir, label_dir, batch_size=16, shuffle=True, transform=None, start_idx=0, end_idx=None):
#     dataset = CustomDataset(image_dir, label_dir, transform=transform, start_idx=start_idx, end_idx=end_idx)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#



# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import numpy as np
#
#
# class CustomDataset(Dataset):
#     def __init__(self, image_dir, label_dir, transform=None, start_idx=0, end_idx=None):
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.image_files = sorted(os.listdir(image_dir))[start_idx:end_idx]
#         self.label_files = sorted(os.listdir(label_dir))[start_idx:end_idx]
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.image_files[idx])
#         label_path = os.path.join(self.label_dir, self.label_files[idx])
#
#         image = Image.open(image_path).convert('RGB')
#         label = Image.open(label_path).convert('P')
#
#         image = image.resize((852, 480))
#         label = label.resize((852, 480))
#
#         if self.transform:
#             image = self.transform(image)
#         else:
#             image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
#
#         label = torch.tensor(np.array(label), dtype=torch.long)
#
#         return image, label
#
#
# def get_dataloader(image_dir, label_dir, batch_size=16, shuffle=True, transform=None, start_idx=0, end_idx=None):
#     dataset = CustomDataset(image_dir, label_dir, transform=transform, start_idx=start_idx, end_idx=end_idx)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import numpy as np
# from torchvision import transforms
#
# class CustomDataset(Dataset):
#     def __init__(self, image_dir, label_dir, transform=None):
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.image_files = sorted(os.listdir(image_dir))
#         self.label_files = sorted(os.listdir(label_dir))
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.image_files[idx])
#         label_path = os.path.join(self.label_dir, self.label_files[idx])
#
#         # 读取原图 (黑白图像)
#         image = Image.open(image_path).convert('RGB')  # 将黑白图像转换为 RGB 模式
#
#         # 读取标记图像
#         label = Image.open(label_path).convert('P')  # 保持调色板模式
#
#         image = image.resize((852, 480))
#         label = label.resize((852, 480))
#
#         if self.transform:
#             image = self.transform(image)
#             label = torch.tensor(np.array(label), dtype=torch.long)
#         else:
#             image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0  # 归一化
#             label = torch.tensor(np.array(label), dtype=torch.long)
#
#         # 确保标签值在 [0, num_classes-1] 范围内
#         label = torch.clamp(label, min=0, max=2)  # 假设有3个类别：0, 1, 2
#
#         return image, label
#
# def get_dataloader(image_dir, label_dir, batch_size=16, shuffle=True, transform=None):
#     dataset = CustomDataset(image_dir, label_dir, transform=transform)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
