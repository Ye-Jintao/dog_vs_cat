import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,
                             QPushButton, QFileDialog, QVBoxLayout,
                             QWidget, QHBoxLayout, QMessageBox)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# 定义模型结构（与 resnet18_with_cbam.py 相同）
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
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att

        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x_spatial = x_channel * spatial_att

        return x_spatial


# 实现 BasicBlock（ResNet-18/34 使用）
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


# 带CBAM的 ResNet-18
class ResNet18WithCBAM(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18WithCBAM, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)

        # 在layer2之后添加CBAM模块
        self.cbam1 = CBAM(64 * BasicBlock.expansion)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)

        # 在layer3之后添加CBAM模块
        self.cbam2 = CBAM(128 * BasicBlock.expansion)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        # 在layer4之前添加CBAM模块
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
        x = self.cbam1(x)  # 添加CBAM模块

        x = self.layer2(x)
        x = self.cbam2(x)  # 添加CBAM模块

        x = self.layer3(x)
        x = self.cbam3(x)  # 添加CBAM模块

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 预测线程
class PredictionThread(QThread):
    prediction_done = pyqtSignal(str, float, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, model, device, image_path):
        super().__init__()
        self.model = model
        self.device = device
        self.image_path = image_path

    def run(self):
        try:
            # 加载并预处理图片
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)

            # 预测
            with torch.no_grad():
                output = self.model(image_tensor)
                probability = output.item()
                prediction = "狗" if probability > 0.5 else "猫"
                confidence = probability if prediction == "狗" else 1 - probability

            self.prediction_done.emit(prediction, confidence, probability)
        except Exception as e:
            self.error_occurred.emit(str(e))


class CatDogApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("猫狗分类器")
        self.setGeometry(100, 100, 800, 600)

        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet18WithCBAM().to(self.device)  # 修改为 ResNet18WithCBAM
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
        # 主部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid black;")
        layout.addWidget(self.image_label)

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
        # 打开文件对话框选择图片
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png)"
        )

        if file_path:
            # 显示图片
            try:
                pixmap = QPixmap(file_path)
                if pixmap.isNull():
                    raise ValueError("无法加载图片文件")

                scaled_pixmap = pixmap.scaled(
                    600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.current_image_path = file_path
                self.result_label.setText("图片已加载，点击'分类'按钮进行预测")
                self.classify_button.setEnabled(True)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载图片失败: {str(e)}")

    def classify_image(self):
        if hasattr(self, 'current_image_path'):
            # 禁用按钮避免重复点击
            self.classify_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.result_label.setText("正在分类中，请稍候...")

            # 创建并启动预测线程
            self.prediction_thread = PredictionThread(
                self.model, self.device, self.current_image_path
            )
            self.prediction_thread.prediction_done.connect(self.on_prediction_done)
            self.prediction_thread.error_occurred.connect(self.on_prediction_error)
            self.prediction_thread.finished.connect(self.on_thread_finished)
            self.prediction_thread.start()

    def on_prediction_done(self, prediction, confidence, probability):
        # 显示结果
        self.result_label.setText(
            f"预测结果: {prediction} (置信度: {confidence * 100:.2f}%)\n"
            f"原始输出值: {probability:.4f}"
        )

        # 在图片上显示结果
        pixmap = self.image_label.pixmap()
        new_pixmap = pixmap.copy()
        painter = QPainter(new_pixmap)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 20))
        painter.drawText(
            20, 40,
            f"{prediction} {confidence * 100:.1f}%"
        )
        painter.end()
        self.image_label.setPixmap(new_pixmap)

    def on_prediction_error(self, error_msg):
        QMessageBox.warning(self, "预测错误", f"分类过程中发生错误:\n{error_msg}")
        self.result_label.setText("分类失败，请重试")

    def on_thread_finished(self):
        # 重新启用按钮
        self.classify_button.setEnabled(True)
        self.load_button.setEnabled(True)

        # 清理线程
        self.prediction_thread = None


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局样式
    app.setStyle("Fusion")

    window = CatDogApp()
    window.show()

    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"应用程序异常退出: {str(e)}")
