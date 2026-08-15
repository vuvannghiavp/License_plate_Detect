"""
color_cnn.py — Kiến trúc mạng CNN nhận màu biển số
====================================================
ColorCNN: 3 conv block + BatchNorm + Dropout
Input : ảnh RGB 64×64
Output: xác suất 4 lớp màu (Đỏ / Trắng / Vàng / Xanh)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorCNN(nn.Module):
    """
    Luồng dữ liệu:
        (N,3,64,64) -> Block1 -> (N,32,32,32)
                    -> Block2 -> (N,64,16,16)
                    -> Block3 -> (N,128,8,8)
                    -> Flatten -> (N,8192)
                    -> FC1+BN+Dropout -> (N,256)
                    -> FC2+BN+Dropout -> (N,128)
                    -> fc_out -> (N,4) logit / xác suất
    """
    def __init__(self, num_classes=4):
        super().__init__()

        # Conv block 1: 3 kênh RGB -> 32 feature maps, pooling 64->32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32) # chuẩn hóa đầu ra
        self.pool1 = nn.MaxPool2d(2, 2)
        #kernel_size=2 kích thước bộ lọc tích chập, stride=2: lấy giá trị lớn nhất trong mỗi ô 2×2 và bước nhảy là 2.
        #Kết quả: giảm kích thước không gian (height, width) xuống một nửa, nhưng giữ nguyên số kênh (128).
        
        # Conv block 2: 32 -> 64 feature maps, pooling 32->16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Conv block 3: 64 -> 128 feature maps, pooling 16->8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully-connected: 128*8*8=8192 -> 256 -> 128 -> num_classes
        self.fc1    = nn.Linear(128 * 8 * 8, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop1  = nn.Dropout(0.4)

        self.fc2    = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.drop2  = nn.Dropout(0.3)

        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))#hàm relu là chuyển từ 1 dãy âm dương thành chỉ có 0 và dương, nếu âm thì chuyển thành 0
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)# chuyển từ 4D(batch, channel, height, width) thành 2D (batch, features).
        x = self.drop1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn_fc2(self.fc2(x))))
        logits = self.fc_out(x)
        # Train -> trả logit thô (CrossEntropyLoss tự tích hợp Softmax)
        # Eval  -> trả xác suất Softmax để đọc confidence
        return F.softmax(logits, dim=1) if not self.training else logits


def load_color_model(path: str, device: torch.device) -> nn.Module:
    """
    Load ColorCNN từ file .pth.
    Trả về model ở eval mode, sẵn sàng inference.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    model = ColorCNN(num_classes=4).to(device)
    model.load_state_dict(state)
    model.eval()
    return model