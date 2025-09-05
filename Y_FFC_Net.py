import torch
import torch.nn as nn
from torchvision import models
from ffc import *
import argparse
import sys

class Conv2D_BN_activa(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding=0,
            dilation=1, if_bn=True, activation='relu', bias=None, initializer=None, transpose=False
    ):
        super(Conv2D_BN_activa, self).__init__()
        if transpose:
            self.conv2d = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation, bias=(not if_bn) if bias is None else bias
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation, bias=(not if_bn) if bias is None else bias
            )
        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)  # eps same as that in the official codes.
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None
        self.initializer = initializer
        if self.initializer is not None:
            if self.initializer == 'truncated_norm':
                nn.init.normal_(self.conv2d.weight, std=0.02)
                self.conv2d.weight = truncated_normal_(self.conv2d.weight, std=0.02)

    def forward(self, x):
        x = self.conv2d(x)
        if self.activation:
            if self.if_bn:
                x = self.bn(x)
            x = self.activation(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class FeatureExtrationLayersA(nn.Module):
    def __init__(self, in_channels, out_channels_branch=192):
        super(FeatureExtrationLayersA, self).__init__()
        self.path1 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 96, 1, 1, 0),
            Conv2D_BN_activa(96, out_channels_branch, 7, 1, 3)
        )
        self.path2 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.path3 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 256, 1, 1, 0),
            Conv2D_BN_activa(256, 256, 3, 1, 1),
            Conv2D_BN_activa(256, out_channels_branch, 3, 1, 1)
        )

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        out = torch.cat((x1, x2, x3), 1)

        return out


class FeatureExtrationLayersB(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtrationLayersB, self).__init__()
        self.path1 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 192, 7, 1, 3)
        )
        self.path2 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 192, 3, 1, 1)
        )
        self.path3 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 128, 3, 1, 1)
        )
        self.path4 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 128, 3, 1, 1),
            Conv2D_BN_activa(128, 128, 3, 1, 1)
        )

    def forward(self, x):
        path1 = self.path1(x)
        path2 = self.path2(x)
        path3 = self.path3(x)
        path4 = self.path4(x)

        out = torch.cat((path1, path2, path3, path4), 1)
        return out


class ImageDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels_branch=256):
        super(ImageDecBlock, self).__init__()
        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        x_123 = torch.cat([x_1, x_2, x_3], dim=1)
        return x_123

class ImageDecBlock_same(nn.Module):
    def __init__(self, in_channels, out_channels_branch=256):
        super(ImageDecBlock_same, self).__init__()

        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        x_123 = torch.cat([x_1, x_2, x_3], dim=1)
        return x_123

class CencN(nn.Module):
    def __init__(self):
        super(CencN, self).__init__()
        backbone_model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        backbone_model_list = list(backbone_model.features.children())
        self.backbone_1 = nn.Sequential(*backbone_model_list[0:7])
        self.backbone_2 = nn.Sequential(*backbone_model_list[7:14])
        self.backbone_3 = nn.Sequential(*backbone_model_list[14:24])
        self.backbone_4 = nn.Sequential(*backbone_model_list[24:34])
        self.backbone_5 = nn.Sequential(*backbone_model_list[34:44])
        self.conv1 = Conv2D_BN_activa(1, 3, 3, 1, 1)
        self.cba_after_backbone = Conv2D_BN_activa(512, 256, 3, 1, 1)

    def forward(self, x):
        x1=self.conv1(x)
        x_bb_1 = self.backbone_1(x1)
        x_bb_2 = self.backbone_2(x_bb_1)
        x_bb_3 = self.backbone_3(x_bb_2)
        x_bb_4 = self.backbone_4(x_bb_3)
        x_c = self.cba_after_backbone(x_bb_4)


        return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c


class GdecN(nn.Module):
    def __init__(self):
        super(GdecN, self).__init__()
        self.block_1_conv_1 = Conv2D_BN_activa(512, 1024, 7, 1, 3)
        self.block_1_conv_2 = Conv2D_BN_activa(1024, 512, 1, 1, 0)
        self.block_1_conv_3 = Conv2D_BN_activa(512, 256, 3, 1, 1)
        self.block_1_conv_transpose = Conv2D_BN_activa(256, 256, 5, 1, 2, transpose=False)

        self.block_2_conv = Conv2D_BN_activa(256 + 512, 128, 3, 1, 1)
        self.block_2_conv_transpose = Conv2D_BN_activa(128, 128, 4, 2, 1, transpose=True)

        self.block_3_conv = Conv2D_BN_activa(128 + 256, 64, 3, 1, 1)
        self.block_3_conv_transpose = Conv2D_BN_activa(64, 64, 4, 2, 1, transpose=True)

        self.block_4_conv = Conv2D_BN_activa(64 + 128, 32, 3, 1, 1)
        self.block_4_conv_transpose = Conv2D_BN_activa(32, 32, 4, 2, 1, transpose=True)

        self.block_5_conv_1 = Conv2D_BN_activa(32 + 64, 64, 4, 2, 1, transpose=True)
        self.block_5_conv_2 = Conv2D_BN_activa(64, 1, 5, 1, 2, activation=None, if_bn=False)

        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, backbone_features):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4 = backbone_features
        x = self.block_1_conv_1(x_bb_4)
        x = self.block_1_conv_2(x)
        x = self.block_1_conv_3(x)
        x_to_be_enhanced_1 = self.block_1_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_1, x_bb_4], dim=1)

        x = self.block_2_conv(x)
        x_to_be_enhanced_2 = self.block_2_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_2, x_bb_3], dim=1)

        x = self.block_3_conv(x)
        x_to_be_enhanced_3 = self.block_3_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_3, x_bb_2], dim=1)

        x = self.block_4_conv(x)
        x_to_be_enhanced_4 = self.block_4_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_4, x_bb_1], dim=1)

        x = self.block_5_conv_1(x)
        x = self.block_5_conv_2(x)

        x_g = self.sigmoid_layer(x)

        return x_to_be_enhanced_1, x_to_be_enhanced_2, x_to_be_enhanced_3, x_to_be_enhanced_4, x_g


class IdecN(nn.Module):
    def __init__(self, in_channels):
        super(IdecN, self).__init__()
        self.feature_extraction_layers_A = FeatureExtrationLayersA(in_channels)
        self.image_dec_block_1 = ImageDecBlock_same(192 * 3, 256)
        self.image_dec_block_2 = ImageDecBlock(256 * 3 + 128 + 512, 128)  # channels of [image_dec_block,
        self.feature_extraction_layers_B = FeatureExtrationLayersB(
            128 * 3 + 64 + 256)  # feature_enhancement_layers, vgg16]
        self.image_dec_block_3 = ImageDecBlock(192 + 192 + 128 + 128, 64)
        self.image_dec_block_4 = ImageDecBlock(64 * 3 + 32 + 128, 32)
        self.image_dec_block_5 = ImageDecBlock(32 * 3 + 16 + 64, 16)
        self.cba_output_1 = Conv2D_BN_activa(16 * 3 + 1, 16, 3, 1, 1)
        self.cba_output_2 = Conv2D_BN_activa(16, 1, 3, 1, 1)
        self.FFC1 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)
        self.FFC2 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)
        self.att1=Attention_block(768,512,640)
        self.att2 = Attention_block(384, 256, 320)
        self.att3 = Attention_block(192, 128, 160)
        self.att4 = Attention_block(96, 64, 80)

    def forward(self, c_features, enhanced_features, x_g):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c = c_features
        x_enhanced_1, x_enhanced_2, x_enhanced_3, x_enhanced_4 = enhanced_features
        x0, x1 = x_c.chunk(2, dim=1)
        x=self.FFC1(tuple([x0,x1]))
        x=self.FFC2(x)
        x = torch.cat([x[0], x[1]], dim=1)
        x=x+x_c
        x = self.feature_extraction_layers_A(x)
        x = self.image_dec_block_1(x)
        bridge=self.att1(x,x_bb_4)
        x = torch.cat([x, bridge,x_enhanced_1], dim=1)

        x = self.image_dec_block_2(x)
        bridge = self.att2(x, x_bb_3)
        x = torch.cat([x, bridge, x_enhanced_2], dim=1)

        x = self.feature_extraction_layers_B(x)
        x = self.image_dec_block_3(x)
        bridge = self.att3(x, x_bb_2)
        x = torch.cat([x, bridge, x_enhanced_3], dim=1)

        x = self.image_dec_block_4(x)
        bridge=self.att4(x, x_bb_1)
        x = torch.cat([x, bridge, x_enhanced_4], dim=1)

        x = self.image_dec_block_5(x)
        x = torch.cat([x, x_g], dim=1)

        x = self.cba_output_1(x)
        est_b = self.cba_output_2(x)

        return est_b


class EnhancementLayers(nn.Module):
    def __init__(self):
        super(EnhancementLayers, self).__init__()
        self.enhancement_layers_1 = Conv2D_BN_activa(256, 128, 7, 1, 3)
        self.enhancement_layers_2 = Conv2D_BN_activa(128, 64, 7, 1, 3)
        self.enhancement_layers_3 = Conv2D_BN_activa(64, 32, 7, 1, 3)
        self.enhancement_layers_4 = Conv2D_BN_activa(32, 16, 7, 1, 3)

    def forward(self, to_be_enhanced_features):
        x_enhanced_1 = self.enhancement_layers_1(to_be_enhanced_features[0])
        x_enhanced_2 = self.enhancement_layers_2(to_be_enhanced_features[1])
        x_enhanced_3 = self.enhancement_layers_3(to_be_enhanced_features[2])
        x_enhanced_4 = self.enhancement_layers_4(to_be_enhanced_features[3])

        return x_enhanced_1, x_enhanced_2, x_enhanced_3, x_enhanced_4


class YFFC(nn.Module):
    def __init__(self):
        super(YFFC, self).__init__()
        self.encoder_context = CencN()
        self.decoder_gradient = GdecN()
        self.enhancement_layers = EnhancementLayers()
        self.decoder_image = IdecN(in_channels=256)


    def forward(self, x):
        c_features = self.encoder_context(x)
        g_features = self.decoder_gradient(c_features[:-1])
        enhanced_features = self.enhancement_layers(g_features[:-1])
        est_b = self.decoder_image(c_features, enhanced_features, g_features[-1])
        estimations = {'r': x - est_b, 'g': g_features[-1], 'b': est_b}
        return estimations

def truncated_normal_(tensor, mean=0.0, std=1.0):
    """
    截断正态分布初始化函数
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def display_model_summary(model, input_size, device='cpu'):
    """
    显示模型结构摘要，优先使用torchinfo，回退到torchsummary
    """
    try:
        from torchinfo import summary
        print("使用 torchinfo 显示模型结构...")
        summary(model, input_size=input_size, device=device, verbose=1)
    except ImportError:
        try:
            from torchsummary import summary
            print("使用 torchsummary 显示模型结构...")
            summary(model, input_size[1:], device=device)  # torchsummary不需要batch维度
        except ImportError:
            print("警告: 未找到 torchinfo 或 torchsummary，无法显示模型结构")
            print("请安装: pip install torchinfo 或 pip install torchsummary")
            return False
    return True

def main():
    """
    主函数：解析命令行参数并显示模型结构
    """
    parser = argparse.ArgumentParser(description='Y-FFC网络模型结构显示工具')
    parser.add_argument('--input_channels', type=int, default=1, 
                       help='输入图像通道数 (默认: 1)')
    parser.add_argument('--image_size', type=int, default=512, 
                       help='输入图像尺寸 (默认: 512)')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='批次大小 (默认: 1)')
    parser.add_argument('--device', type=str, default='cpu', 
                       choices=['cpu', 'cuda'], help='运行设备 (默认: cpu)')
    parser.add_argument('--show_params', action='store_true', 
                       help='显示详细参数统计')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Y-FFC网络模型结构显示")
    print("=" * 60)
    print(f"输入通道数: {args.input_channels}")
    print(f"图像尺寸: {args.image_size}x{args.image_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"运行设备: {args.device}")
    print("=" * 60)
    
    # 创建模型
    try:
        model = YFFC()
        print("✓ Y-FFC模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return
    
    # 设置设备
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU")
        device = torch.device('cpu')
    
    model.to(device)
    print(f"✓ 模型已移动到设备: {device}")
    
    # 准备输入数据
    input_size = (args.batch_size, args.input_channels, args.image_size, args.image_size)
    test_input = torch.randn(input_size).to(device)
    print(f"✓ 测试输入数据生成成功，形状: {test_input.shape}")
    
    # 测试前向传播
    try:
        with torch.no_grad():
            output = model(test_input)
            print("\n" + "=" * 30)
            print("模型前向传播测试成功!")
            print("=" * 30)
            print(f"输入形状: {test_input.shape}")
            print(f"漫反射输出形状: {output['r'].shape}")
            print(f"梯度输出形状: {output['g'].shape}")
            print(f"镜面反射输出形状: {output['b'].shape}")
            
            # 验证输出形状
            assert output['r'].shape == test_input.shape, "漫反射输出形状不匹配"
            assert output['g'].shape == test_input.shape, "梯度输出形状不匹配"
            assert output['b'].shape == test_input.shape, "镜面反射输出形状不匹配"
            print("✓ 所有输出形状验证通过")
    except Exception as e:
        print(f"\n❌ 模型测试失败: {e}")
        return
    
    # 显示模型结构
    print("\n" + "=" * 60)
    print("模型结构摘要")
    print("=" * 60)
    
    if not display_model_summary(model, input_size, device):
        return
    
    # 显示参数统计
    if args.show_params:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n参数统计:")
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
    
    print("\n" + "=" * 60)
    print("模型结构显示完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()