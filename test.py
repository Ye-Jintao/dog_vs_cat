import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,
                             QPushButton, QFileDialog, QVBoxLayout,
                             QWidget, QHBoxLayout, QMessageBox)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2

# CBAM模块（带注意力权重输出）
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x_spatial = x_channel * spatial_att

        return x_spatial, channel_att, spatial_att  # 返回注意力权重


# BasicBlock实现
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 带注意力机制的 ResNet-18 模型
class ResNet18WithCBAM(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18WithCBAM, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)

        self.cbam1 = CBAM(64 * BasicBlock.expansion)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)

        self.cbam2 = CBAM(128 * BasicBlock.expansion)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        self.cbam3 = CBAM(256 * BasicBlock.expansion)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x, channel_att1, spatial_att1 = self.cbam1(x)

        x = self.layer2(x)
        x, channel_att2, spatial_att2 = self.cbam2(x)

        x = self.layer3(x)
        x, channel_att3, spatial_att3 = self.cbam3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, {
            'channel': [channel_att1, channel_att2, channel_att3],
            'spatial': [spatial_att1, spatial_att2, spatial_att3]
        }


# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 预测线程（含注意力图）
class PredictionThread(QThread):
    prediction_done = pyqtSignal(str, float, float, dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, model, device, image_path):
        super().__init__()
        self.model = model
        self.device = device
        self.image_path = image_path

    def run(self):
        try:
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output, attentions = self.model(image_tensor)
                probability = output.item()
                prediction = "狗" if probability > 0.5 else "猫"
                confidence = probability if prediction == "狗" else 1 - probability

            self.prediction_done.emit(prediction, confidence, probability, attentions)
        except Exception as e:
            self.error_occurred.emit(str(e))


class CatDogApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("猫狗分类器")
        self.setGeometry(100, 100, 800, 650)

        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet18WithCBAM().to(self.device)
        try:
            self.model.load_state_dict(torch.load('resnet18_CBAM.pth', map_location=self.device))
            self.model.eval()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            sys.exit(1)

        # 创建UI
        self.initUI()
        self.prediction_thread = None

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid black;")
        layout.addWidget(self.image_label)

        # 注意力图显示区域
        self.attn_label = QLabel("注意力热力图")
        self.attn_label.setAlignment(Qt.AlignCenter)
        self.attn_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.attn_label)

        # 结果标签
        self.result_label = QLabel("请选择一张图片进行分类")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.result_label)

        # 按钮区域
        button_layout = QHBoxLayout()

        self.load_button = QPushButton("选择图片")
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setStyleSheet("font-size: 16px; padding: 10px;")
        button_layout.addWidget(self.load_button)

        self.classify_button = QPushButton("分类")
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setStyleSheet("font-size: 16px; padding: 10px;")
        self.classify_button.setEnabled(False)
        button_layout.addWidget(self.classify_button)

        layout.addLayout(button_layout)

        central_widget.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png)")
        if file_path:
            try:
                pixmap = QPixmap(file_path)
                if pixmap.isNull():
                    raise ValueError("无法加载图片文件")

                scaled_pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.current_image_path = file_path
                self.result_label.setText("图片已加载，点击'分类'按钮进行预测")
                self.classify_button.setEnabled(True)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载图片失败: {str(e)}")

    def classify_image(self):
        if hasattr(self, 'current_image_path'):
            self.classify_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.result_label.setText("正在分类中，请稍候...")

            self.prediction_thread = PredictionThread(self.model, self.device, self.current_image_path)
            self.prediction_thread.prediction_done.connect(self.on_prediction_done)
            self.prediction_thread.error_occurred.connect(self.on_prediction_error)
            self.prediction_thread.finished.connect(self.on_thread_finished)
            self.prediction_thread.start()

    def on_prediction_done(self, prediction, confidence, probability, attentions):
        self.result_label.setText(
            f"预测结果: {prediction} (置信度: {confidence * 100:.2f}%)\n"
            f"原始输出值: {probability:.4f}"
        )

        # 加载原始图像
        original_image = cv2.imread(self.current_image_path)
        if original_image is None:
            self.result_label.setText("错误: 无法加载原始图像")
            return
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(original_image, (224, 224))

        # 获取最后一个空间注意力图
        spatial_attn_map = attentions['spatial'][-1].squeeze().cpu().numpy()
        print(
            f"Spatial Attention Map Shape: {spatial_attn_map.shape}, Min: {spatial_attn_map.min()}, Max: {spatial_attn_map.max()}")

        # 检查注意力图尺寸
        if spatial_attn_map.shape[0] == 0 or spatial_attn_map.shape[1] == 0:
            self.result_label.setText("错误: 注意力图尺寸无效")
            return

        # 应用阈值以突出高注意力区域
        threshold = np.percentile(spatial_attn_map, 75)
        spatial_attn_map = np.clip(spatial_attn_map, threshold, spatial_attn_map.max())
        spatial_attn_map = (spatial_attn_map - spatial_attn_map.min()) / (
                    spatial_attn_map.max() - spatial_attn_map.min() + 1e-8)

        # 调整注意力图大小以匹配原始图像
        try:
            heatmap = cv2.resize(spatial_attn_map, (224, 224), interpolation=cv2.INTER_LINEAR)
            alpha_map = cv2.resize(spatial_attn_map, (224, 224), interpolation=cv2.INTER_LINEAR)  # Resize alpha_map
            alpha_map = alpha_map * 0.6  # Apply transparency factor
            alpha_map = np.stack([alpha_map] * 3, axis=-1)  # Add channel dimension to match (224, 224, 3)
        except Exception as e:
            self.result_label.setText(f"错误: 调整注意力图大小失败: {str(e)}")
            return

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 动态透明度叠加
        try:
            overlaid_image = (alpha_map * heatmap + (1 - alpha_map) * original_image).astype(np.uint8)
        except Exception as e:
            self.result_label.setText(f"错误: 图像叠加失败: {str(e)}")
            return

        # 转换为QImage
        height, width, channel = overlaid_image.shape
        bytes_per_line = width * channel
        try:
            qimg = QImage(overlaid_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            if qimg.isNull():
                self.result_label.setText("错误: 无法创建QImage")
                return
            pixmap = QPixmap.fromImage(qimg).scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.attn_label.setPixmap(pixmap)
        except Exception as e:
            self.result_label.setText(f"错误: QImage转换失败: {str(e)}")
            return

        # 保存所有层的注意力图用于调试
        for i, spatial_att in enumerate(attentions['spatial']):
            spatial_map = spatial_att.squeeze().cpu().numpy()
            spatial_map = (spatial_map - spatial_map.min()) / (spatial_map.max() - spatial_map.min() + 1e-8)
            heatmap = cv2.resize(spatial_map, (224, 224), interpolation=cv2.INTER_LINEAR)
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            cv2.imwrite(f"spatial_attention_layer_{i + 1}.png", heatmap)

    def on_prediction_error(self, error_msg):
        QMessageBox.warning(self, "预测错误", f"分类过程中发生错误:\n{error_msg}")
        self.result_label.setText("分类失败，请重试")

    def on_thread_finished(self):
        self.classify_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.prediction_thread = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CatDogApp()
    window.show()
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"应用程序异常退出: {str(e)}")