import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import os

def predict_and_visualize(model, data_loader, device):
    model.eval()
    os.makedirs('results', exist_ok=True)
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            image_index = i + 201  # 确保图片索引从201开始
            print(f"Processing image {image_index}...")  # Debug information
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            preds = torch.argmax(outputs, 1).cpu().numpy()[0]

            image = images.cpu().numpy()[0].transpose(1, 2, 0) * 255.0
            image = image.astype(np.uint8)
            pred_image = np.zeros_like(image)
            label_image = labels.cpu().numpy()[0]

            color_map = {
                0: [0, 0, 0],       # 背景
                1: [128, 0, 0],      # 煤
                2: [0, 128, 0],     # 矸石

            }

            for cls, color in color_map.items():
                pred_image[(preds == cls)] = color

            # 创建 comparison 图像
            label_image_rgb = np.zeros_like(image)
            for cls, color in color_map.items():
                label_image_rgb[(label_image == cls)] = color
            label_image_rgb = Image.fromarray(label_image_rgb)

            pred_image = Image.fromarray(pred_image)
            result_image = Image.fromarray(image)
            draw = ImageDraw.Draw(result_image)

            coal_count = np.sum(preds == 1)
            gangue_count = np.sum(preds == 2)
            total_count = coal_count + gangue_count  # 只计算煤和矸石的总数
            coal_ratio = coal_count / total_count if total_count > 0 else 0

            for y in range(0, preds.shape[0], 20):
                for x in range(0, preds.shape[1], 20):
                    if preds[y, x] == 1:
                        draw.rectangle([x, y, x+20, y+20], outline=(128, 0, 0))
                    elif preds[y, x] == 2:
                        draw.rectangle([x, y, x+20, y+20], outline=(0, 128, 0))

            processing_time = end_time - start_time
            result_image = result_image.resize((852, 480))
            pred_image = pred_image.resize((852, 480))
            label_image_rgb = label_image_rgb.resize((852, 480))

            # 加载字体并设置大小
            font = ImageFont.truetype("arial.ttf", 24)

            # 在 comparison 图的左上角标出处理时长和煤的占比
            comparison_image = Image.new('RGB', (852 * 3, 480))
            comparison_image.paste(result_image, (0, 0))
            comparison_image.paste(pred_image, (852, 0))
            comparison_image.paste(label_image_rgb, (852 * 2, 0))
            draw_comparison = ImageDraw.Draw(comparison_image)
            draw_comparison.text((10, 10), f'Processing time: {processing_time:.4f}s', fill=(255, 255, 255), font=font)
            draw_comparison.text((10, 40), f'Coal ratio: {coal_ratio:.2%}', fill=(255, 255, 255), font=font)

            comparison_path = f'results/{image_index}.png'

            print(f"Saving comparison image to {comparison_path}")  # Debug information

            comparison_image.save(comparison_path)





# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
# import time
# import os
#
# color_map = {
#     0: (0, 0, 0),       # 黑色 - 背景
#     1: (128, 0, 0),     # 红色 - 煤
#     2: (0, 128, 0)      # 绿色 - 矸石
# }
#
# def predict_and_visualize(model, test_loader, device='cpu'):
#     model.to(device)
#     model.eval()
#
#     # 确保结果目录存在
#     if not os.path.exists('results'):
#         os.makedirs('results')
#
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(test_loader):
#             start_time = time.time()
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#
#             _, preds = torch.max(outputs, 1)
#             preds = preds.cpu().numpy()[0]
#             labels = labels.cpu().numpy()[0]
#
#             # 绘制对比图像
#             comparison_image = np.zeros((480, 852, 3), dtype=np.uint8)
#             for cls, color in color_map.items():
#                 comparison_image[labels == cls] = color
#             pred_image = np.zeros((480, 852, 3), dtype=np.uint8)
#             for cls, color in color_map.items():
#                 pred_image[preds == cls] = color
#
#             comparison_image = np.concatenate((comparison_image, pred_image), axis=1)
#
#             # 计算准确率
#             accuracy = np.mean(preds == labels)
#
#             # 绘制格子
#             result_image = images.cpu().numpy()[0].transpose(1, 2, 0) * 255
#             result_image = result_image.astype(np.uint8)
#             result_image = Image.fromarray(result_image)
#             draw = ImageDraw.Draw(result_image)
#
#             # 煤的占比
#             coal_ratio = np.sum(preds == 1) / (np.sum(preds == 1) + np.sum(preds == 2))
#
#             for x in range(0, 852, 20):
#                 for y in range(0, 480, 20):
#                     sub_image = preds[y:y+20, x:x+20]
#                     if np.any(sub_image == 1):  # 煤
#                         draw.rectangle([x, y, x+20, y+20], outline='red')
#                     elif np.any(sub_image == 2):  # 矸石
#                         draw.rectangle([x, y, x+20, y+20], outline='green')
#
#             end_time = time.time()
#             processing_time = end_time - start_time
#
#             # 在图像上标注信息
#             draw.text((10, 10), f'Processing time: {processing_time:.2f}s', fill='yellow')
#             draw.text((10, 30), f'Coal ratio: {coal_ratio:.2f}', fill='yellow')
#
#             # 保存结果图像
#             comparison_image = Image.fromarray(comparison_image)
#             comparison_image.save(f'results/comparison_{i + 236}.png')
#             result_image.save(f'results/prediction_{i + 236}.png')
#
#             print(f'Image {i + 201} processed in {processing_time:.2f} seconds, accuracy: {accuracy:.4f}, coal ratio: {coal_ratio:.4f}')



# import os
# import torch
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
# import time
#
# def predict_and_visualize(model, dataloader, device, output_dir='result'):
#     os.makedirs(output_dir, exist_ok=True)
#     model.eval()
#     total_time = 0
#     correct_pixels = 0
#     total_pixels = 0
#     for i, (images, labels) in enumerate(dataloader):
#         images, labels = images.to(device), labels.to(device)
#         start_time = time.time()
#         with torch.no_grad():
#             outputs = model(images)
#         elapsed_time = time.time() - start_time
#         total_time += elapsed_time
#
#         _, preds = torch.max(outputs, 1)
#         correct_pixels += (preds == labels).sum().item()
#         total_pixels += labels.numel()
#
#         for j in range(images.size(0)):
#             img_idx = i * images.size(0) + j + 201
#             img = images[j].cpu().permute(1, 2, 0).numpy() * 255.0
#             label = labels[j].cpu().numpy()
#             pred = preds[j].cpu().numpy()
#
#             # Create comparison image
#             comparison_image = np.zeros((480, 852, 3), dtype=np.uint8)
#             comparison_image[label == 0] = [0, 0, 0]       # Background
#             comparison_image[label == 1] = [255, 0, 0]     # Coal
#             comparison_image[label == 2] = [0, 255, 0]     # Gangue
#
#             # Create predicted image
#             pred_image = np.zeros((480, 852, 3), dtype=np.uint8)
#             pred_image[pred == 0] = [0, 0, 0]             # Background
#             pred_image[pred == 1] = [255, 0, 0]           # Coal
#             pred_image[pred == 2] = [0, 255, 0]           # Gangue
#
#             # Calculate coal ratio
#             coal_ratio = np.sum(pred == 1) / (np.sum(pred == 1) + np.sum(pred == 2))
#
#             # Create image with bounding boxes
#             result_image = Image.fromarray((img).astype(np.uint8))
#             draw = ImageDraw.Draw(result_image)
#             for y in range(0, 480, 20):
#                 for x in range(0, 852, 20):
#                     patch = pred[y:y+20, x:x+20]
#                     if np.any(patch == 1):  # Coal
#                         draw.rectangle([x, y, x + 20, y + 20], outline="red")
#                     if np.any(patch == 2):  # Gangue
#                         draw.rectangle([x, y, x + 20, y + 20], outline="green")
#
#             # Save comparison image
#             comp_img = Image.fromarray(np.hstack((comparison_image, pred_image)))
#             comp_draw = ImageDraw.Draw(comp_img)
#             comp_draw.text((10, 10), f"Accuracy: {correct_pixels / total_pixels:.4f}", fill="white")
#
#             # Save images
#             comp_img.save(os.path.join(output_dir, f"comparison_{img_idx}.png"))
#             result_image.save(os.path.join(output_dir, f"result_{img_idx}.png"))
#
#             # Print processing time and coal ratio on result image
#             result_draw = ImageDraw.Draw(result_image)
#             result_draw.text((10, 10), f"Processing Time: {elapsed_time:.2f}s", fill="white")
#             result_draw.text((10, 30), f"Coal Ratio: {coal_ratio:.4f}", fill="white")
#             result_image.save(os.path.join(output_dir, f"result_{img_idx}_annotated.png"))
#
#     print(f"Overall Accuracy: {correct_pixels / total_pixels:.4f}")
#     print(f"Average Processing Time: {total_time / total_pixels:.4f}s per image")





# import os
# import torch
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
# import time
#
# def predict_and_visualize(model, dataloader, device, output_dir='result'):
#     os.makedirs(output_dir, exist_ok=True)
#     model.eval()
#     total_time = 0
#     correct_pixels = 0
#     total_pixels = 0
#     for i, (images, labels) in enumerate(dataloader):
#         images, labels = images.to(device), labels.to(device)
#         start_time = time.time()
#         with torch.no_grad():
#             outputs = model(images)
#         elapsed_time = time.time() - start_time
#         total_time += elapsed_time
#
#         _, preds = torch.max(outputs, 1)
#         correct_pixels += (preds == labels).sum().item()
#         total_pixels += labels.numel()
#
#         for j in range(images.size(0)):
#             img_idx = i * images.size(0) + j + 201
#             img = images[j].cpu().permute(1, 2, 0).numpy() * 255.0
#             label = labels[j].cpu().numpy()
#             pred = preds[j].cpu().numpy()
#
#             # Create comparison image
#             comparison_image = np.zeros((480, 852, 3), dtype=np.uint8)
#             comparison_image[label == 0] = [0, 0, 0]       # Background
#             comparison_image[label == 1] = [255, 0, 0]     # Coal
#             comparison_image[label == 2] = [0, 255, 0]     # Gangue
#
#             # Create predicted image
#             pred_image = np.zeros((480, 852, 3), dtype=np.uint8)
#             pred_image[pred == 0] = [0, 0, 0]             # Background
#             pred_image[pred == 1] = [255, 0, 0]           # Coal
#             pred_image[pred == 2] = [0, 255, 0]           # Gangue
#
#             # Calculate coal ratio
#             coal_ratio = np.sum(pred == 1) / (np.sum(pred == 1) + np.sum(pred == 2))
#
#             # Create image with bounding boxes
#             result_image = Image.fromarray((img).astype(np.uint8))
#             draw = ImageDraw.Draw(result_image)
#             for y in range(0, 480, 20):
#                 for x in range(0, 852, 20):
#                     patch = pred[y:y+20, x:x+20]
#                     if np.any(patch == 1):  # Coal
#                         draw.rectangle([x, y, x + 20, y + 20], outline="red")
#                     if np.any(patch == 2):  # Gangue
#                         draw.rectangle([x, y, x + 20, y + 20], outline="green")
#
#             # Save comparison image
#             comp_img = Image.fromarray(np.hstack((comparison_image, pred_image)))
#             comp_draw = ImageDraw.Draw(comp_img)
#             comp_draw.text((10, 10), f"Accuracy: {correct_pixels / total_pixels:.4f}", fill="white")
#
#             # Save images
#             comp_img.save(os.path.join(output_dir, f"comparison_{img_idx}.png"))
#             result_image.save(os.path.join(output_dir, f"result_{img_idx}.png"))
#
#             # Print processing time and coal ratio on result image
#             result_draw = ImageDraw.Draw(result_image)
#             result_draw.text((10, 10), f"Processing Time: {elapsed_time:.2f}s", fill="white")
#             result_draw.text((10, 30), f"Coal Ratio: {coal_ratio:.4f}", fill="white")
#             result_image.save(os.path.join(output_dir, f"result_{img_idx}_annotated.png"))
#
#     print(f"Overall Accuracy: {correct_pixels / total_pixels:.4f}")
#     print(f"Average Processing Time: {total_time / total_pixels:.4f}s per image")
#
