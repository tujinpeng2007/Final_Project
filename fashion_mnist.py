"""
修复版GPU训练代码 - 解决Windows多进程问题
强制使用GPU，包含所有必需功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import os
import warnings

warnings.filterwarnings('ignore')


def main():
    # ============================
    # 1. GPU强制配置
    # ============================
    print("=" * 60)
    print("GPU强制配置")
    print("=" * 60)

    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("错误: 未检测到GPU!")
        print("请确保:")
        print(
            "1. 安装了GPU版本的PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("2. 有NVIDIA显卡并安装了CUDA")
        print("3. 安装了最新的NVIDIA驱动")
        raise RuntimeError("需要GPU来运行此代码!")

    # 设置GPU设备
    device = torch.device("cuda:0")
    print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA版本: {torch.version.cuda}")
    print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # 优化GPU设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # 清理GPU缓存
    torch.cuda.empty_cache()

    # 设置随机种子
    def set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    # ============================
    # 2. 数据加载和预处理
    # ============================
    print("\n" + "=" * 60)
    print("数据加载和预处理")
    print("=" * 60)

    # 数据增强
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载数据集
    print("下载Fashion-MNIST数据集...")
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # GPU优化设置 - 减少工作进程避免Windows多进程问题
    batch_size = 256
    num_workers = 0  # Windows下设为0避免多进程问题
    pin_memory = True

    print(f"\n数据统计:")
    print(f"训练集: {len(train_dataset):,} 样本")
    print(f"验证集: {len(val_dataset):,} 样本")
    print(f"测试集: {len(test_dataset):,} 样本")
    print(f"批次大小: {batch_size}")
    print(f"工作进程: {num_workers} (Windows设为0避免多进程错误)")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # 类别名称
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # ============================
    # 3. 构建深度卷积神经网络
    # ============================
    print("\n" + "=" * 60)
    print("构建深度卷积神经网络")
    print("=" * 60)

    class DeepFashionCNN(nn.Module):
        """深度CNN模型，专门为GPU优化设计"""

        def __init__(self):
            super(DeepFashionCNN, self).__init__()

            # 特征提取器 - 3个卷积块
            self.features = nn.Sequential(
                # 卷积块1
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),

                # 卷积块2
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),

                # 卷积块3
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25)
            )

            # 分类器 - 3个全连接层
            self.classifier = nn.Sequential(
                nn.Linear(256 * 3 * 3, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),

                nn.Linear(128, 10)  # 输出层
            )

            # 初始化权重
            self._initialize_weights()

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)  # 展平
            x = self.classifier(x)
            return x

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    # 创建模型并移动到GPU
    model = DeepFashionCNN().to(device)
    print("模型架构:")
    print(model)
    print(f"\n模型统计:")
    print(f"总参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 如果有多GPU，使用数据并行
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 个GPU，启用数据并行")
        model = nn.DataParallel(model)

    # ============================
    # 4. 训练配置
    # ============================
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)

    # 使用混合精度训练加速（新API）
    scaler = torch.amp.GradScaler('cuda')

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 动态学习率调度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=20,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    print("训练配置:")
    print(f"损失函数: CrossEntropyLoss with label smoothing")
    print(f"优化器: AdamW (lr=0.001, weight_decay=1e-4)")
    print(f"学习率调度: OneCycleLR")
    print(f"混合精度训练: 启用")
    print(f"批次大小: {batch_size}")

    # ============================
    # 5. 训练函数
    # ============================
    def train_epoch_gpu(epoch):
        """GPU训练一个epoch"""
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 将数据移动到GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # 使用自动混合精度
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 更新学习率
            scheduler.step()

            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 每50个batch显示一次进度
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx:3d}/{len(train_loader):3d} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {100. * predicted.eq(targets).sum().item() / targets.size(0):.1f}%")

        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate_gpu():
        """GPU验证"""
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc

    # ============================
    # 6. 训练循环
    # ============================
    print("\n" + "=" * 60)
    print("开始GPU训练")
    print("=" * 60)

    num_epochs = 20
    best_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    learning_rates = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # 训练
        train_loss, train_acc = train_epoch_gpu(epoch + 1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        val_loss, val_acc = validate_gpu()
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 记录学习率
        learning_rates.append(optimizer.param_groups[0]['lr'])

        epoch_time = time.time() - epoch_start

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'best_acc': best_acc,
            }, 'best_model_gpu.pt')
            print(f"  ✓ 保存最佳模型，验证准确率: {val_acc:.2f}%")

        print(f"\n  Epoch Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {learning_rates[-1]:.6f}")
        print("-" * 40)

    total_time = time.time() - start_time
    print(f"\n训练完成! 总时间: {total_time:.1f}秒")
    print(f"最佳验证准确率: {best_acc:.2f}%")

    # ============================
    # 7. 可视化训练过程
    # ============================
    print("\n" + "=" * 60)
    print("可视化训练过程")
    print("=" * 60)

    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', linewidth=2, label='Train Loss')
    plt.plot(val_losses, 'r-', linewidth=2, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, 'b-', linewidth=2, label='Train Acc')
    plt.plot(val_accs, 'r-', linewidth=2, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=best_acc, color='g', linestyle='--', alpha=0.5, label=f'Best: {best_acc:.1f}%')

    # 学习率曲线
    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)

    plt.suptitle(f'GPU Training Results (Best Val Acc: {best_acc:.2f}%, Total Time: {total_time:.1f}s)', fontsize=14)
    plt.tight_layout()
    plt.savefig('gpu_training_results.png', dpi=120, bbox_inches='tight')
    plt.show()

    # ============================
    # 8. 加载最佳模型并测试
    # ============================
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)

    # 加载最佳模型
    checkpoint = torch.load('best_model_gpu.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            # 显示进度
            if batch_idx % 10 == 0:
                print(f"  Testing batch {batch_idx}/{len(test_loader)}")

    test_acc = 100. * correct / total
    test_loss = test_loss / len(test_loader)

    print(f"\n测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"最佳验证准确率: {best_acc:.2f}%")

    # ============================
    # 9. 混淆矩阵
    # ============================
    print("\n生成混淆矩阵...")
    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - GPU Training (Test Acc: {test_acc:.2f}%)', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix_gpu.png', dpi=120, bbox_inches='tight')
    plt.show()

    # 分类报告
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_predictions,
                                target_names=class_names, digits=3))

    # ============================
    # 10. 可视化预测结果
    # ============================
    print("\n" + "=" * 60)
    print("可视化预测结果")
    print("=" * 60)

    def visualize_predictions():
        model.eval()
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        images, labels = images[:15], labels[:15]

        with torch.no_grad():
            images_gpu = images.to(device)
            outputs = model(images_gpu)
            _, predictions = outputs.max(1)
            predictions = predictions.cpu()

        fig, axes = plt.subplots(3, 5, figsize=(15, 9))

        for idx in range(15):
            ax = axes[idx // 5, idx % 5]
            img = images[idx].squeeze().numpy()

            ax.imshow(img, cmap='gray')

            true_label = class_names[labels[idx].item()]
            pred_label = class_names[predictions[idx].item()]

            if labels[idx] == predictions[idx]:
                color = 'green'
                result = "✓"
            else:
                color = 'red'
                result = "✗"

            ax.set_title(f"True: {true_label}\nPred: {pred_label} {result}",
                         color=color, fontsize=9)
            ax.axis('off')

        plt.suptitle(f'Model Predictions on Test Set (Green: Correct, Red: Wrong)\nTest Accuracy: {test_acc:.2f}%',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig('gpu_predictions.png', dpi=120, bbox_inches='tight')
        plt.show()

    visualize_predictions()

    # ============================
    # 11. 保存最终模型和结果
    # ============================
    print("\n" + "=" * 60)
    print("保存最终结果")
    print("=" * 60)

    # 保存完整模型
    final_model_path = 'fashion_mnist_gpu_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'test_accuracy': test_acc,
        'best_accuracy': best_acc,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    }, final_model_path)

    # 保存训练结果报告
    with open('training_report_gpu.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Fashion-MNIST GPU Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Epochs: {num_epochs}\n")
        f.write(f"  Model: DeepFashionCNN\n")
        f.write(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")

        f.write(f"Training Results:\n")
        f.write(f"  Total Training Time: {total_time:.1f} seconds\n")
        f.write(f"  Best Validation Accuracy: {best_acc:.2f}%\n")
        f.write(f"  Final Test Accuracy: {test_acc:.2f}%\n\n")

        f.write(f"Per-class Accuracy:\n")
        report = classification_report(all_labels, all_predictions,
                                       target_names=class_names, digits=3, output_dict=True)
        for class_name in class_names:
            f.write(f"  {class_name}: {report[class_name]['precision'] * 100:.1f}%\n")

    print("已保存的文件:")
    print(f"  1. best_model_gpu.pt - 最佳模型")
    print(f"  2. fashion_mnist_gpu_final.pt - 最终完整模型")
    print(f"  3. gpu_training_results.png - 训练曲线图")
    print(f"  4. confusion_matrix_gpu.png - 混淆矩阵")
    print(f"  5. gpu_predictions.png - 预测可视化")
    print(f"  6. training_report_gpu.txt - 训练报告")

    # ============================
    # 12. 性能统计
    # ============================
    print("\n" + "=" * 60)
    print("GPU性能统计")
    print("=" * 60)

    # GPU内存使用
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        gpu_memory_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
        print(f"GPU内存使用:")
        print(f"  已分配: {gpu_memory:.2f} GB")
        print(f"  已保留: {gpu_memory_reserved:.2f} GB")
        print(f"  峰值使用率: {gpu_memory / torch.cuda.get_device_properties(0).total_memory * 1024 ** 3:.1f}%")

    print(f"\n训练速度:")
    print(f"  总时间: {total_time:.1f} 秒")
    print(f"  平均每轮: {total_time / num_epochs:.1f} 秒")
    print(f"  每秒样本数: {len(train_dataset) * num_epochs / total_time:.0f}")

    print(f"\n模型性能:")
    print(f"  最佳验证准确率: {best_acc:.2f}%")
    print(f"  最终测试准确率: {test_acc:.2f}%")
    print(f"  准确率提升: {test_acc - 90:.2f}% (相对于基准)")

    print("\n" + "=" * 60)
    print("GPU训练完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()