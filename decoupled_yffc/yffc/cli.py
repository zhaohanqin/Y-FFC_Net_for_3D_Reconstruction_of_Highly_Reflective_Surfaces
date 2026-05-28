"""CLI compatible with the original Y_FFC_unified.py script."""

import argparse

import torch
import torch.nn as nn

from .losses import CompositeLoss
from .models import YFFC
from .training.engine import train_model
from .training.utils import EMA, replace_bn_with_gn


def display_basic_model_info(model):
    print("\n" + "=" * 60)
    print("模型层次结构")
    print("=" * 60)
    print("\n【主要模块】")
    for name, module in model.named_children():
        print(f"  * {name}: {module.__class__.__name__}")
    print("\n【详细结构】")
    for name, module in model.named_children():
        print(f"\n{name} ({module.__class__.__name__}):")
        for sub_name, sub_module in module.named_children():
            print(f"  -- {sub_name}: {sub_module.__class__.__name__}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数: {total_params:,}, 可训练: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print("\n【各模块参数量】")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        pct = (module_params / total_params * 100) if total_params > 0 else 0
        print(f"  * {name}: {module_params:,} ({pct:.1f}%)")


def display_model_summary(model, input_size, device="cpu"):
    try:
        from torchinfo import summary as torchinfo_summary
        torchinfo_summary(model, input_size=input_size, device=device, verbose=1)
        return True
    except ImportError:
        try:
            from torchsummary import summary as torchsummary_summary
            torchsummary_summary(model, input_size=input_size[1:], device=device)
            return True
        except ImportError:
            display_basic_model_info(model)
            return False


def main():
    parser = argparse.ArgumentParser(description="Y-FFC 统一版——网络测试与训练")
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # 子命令 1: test
    test_parser = subparsers.add_parser('test', help='网络前向测试')
    test_parser.add_argument("--input_channels", type=int, default=1)
    test_parser.add_argument("--image_size", type=int, default=512)
    test_parser.add_argument("--batch_size", type=int, default=1)
    test_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    test_parser.add_argument("--show_params", action="store_true", default=True)
    test_parser.add_argument("--show_summary", action="store_true", default=True)

    # 子命令 2: train
    train_parser = subparsers.add_parser('train', help='模型训练')
    train_parser.add_argument("--input_dir", type=str, required=True)
    train_parser.add_argument("--target_dir", type=str, required=True)
    train_parser.add_argument("--val_input_dir", type=str, default=None)
    train_parser.add_argument("--val_target_dir", type=str, default=None)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch_size", type=int, default=2)
    train_parser.add_argument("--image_size", type=int, default=512)
    train_parser.add_argument("--train_split_ratio", type=float, default=0.8)
    train_parser.add_argument("--base_lr", type=float, default=2e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--accumulation_steps", type=int, default=4)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--save_dir", type=str, default="./checkpoints")
    train_parser.add_argument("--save_interval", type=int, default=10)
    train_parser.add_argument("--use_gn", action="store_true")
    train_parser.add_argument("--use_ema", action="store_true")
    train_parser.add_argument("--ema_decay", type=float, default=0.9995)
    train_parser.add_argument("--use_ohem", action="store_true", default=True)
    train_parser.add_argument("--no_ohem", action="store_true")
    train_parser.add_argument("--use_augmentation", action="store_true", default=True)
    train_parser.add_argument("--no_augmentation", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        args.command = 'test'
        args.input_channels = 1
        args.image_size = 512
        args.batch_size = 1
        args.device = 'cpu'
        args.show_params = True
        args.show_summary = True

    if args.command == 'test':
        print("=" * 60)
        print("Y-FFC 统一版 网络前向测试")
        print("=" * 60)

        device = torch.device(args.device)
        if args.device == "cuda" and not torch.cuda.is_available():
            print("警告: CUDA 不可用，切换到 CPU")
            device = torch.device("cpu")

        # 创建模型
        model = YFFC()
        print("测试模型: YFFC")
        model.to(device)

        input_size = (args.batch_size, args.input_channels, args.image_size, args.image_size)
        test_input = torch.randn(input_size).to(device)

        # 前向传播测试
        try:
            with torch.no_grad():
                out = model(test_input)
            print(f"前向传播成功: r={out['r'].shape}, g={out['g'].shape}, "
                  f"b={out['b'].shape}, s={out['s'].shape}")
            assert out["r"].shape == test_input.shape
            assert out["g"].shape == test_input.shape
            assert out["b"].shape == test_input.shape
            assert out["s"].shape == test_input.shape
            print("输出 shape 校验通过")
        except Exception as e:
            print(f"前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 损失函数测试
        print("\n--- 损失函数测试 ---")
        try:
            criterion = CompositeLoss(use_perceptual=False).to(device)
            target = torch.rand_like(out['r'])
            gt_specular = (test_input - target).clamp(0, 1)
            gt_gradient = torch.rand_like(out['g'])
            with torch.no_grad():
                loss, loss_dict = criterion(
                    out['r'], target, out['b'], gt_specular,
                    out['g'], gt_gradient, pred_s_lik=out['s'])
            print(f"CompositeLoss 前向成功, total={loss.item():.4f}, "
                  f"keys={list(loss_dict.keys())}")
        except Exception as e:
            print(f"损失函数测试失败: {e}")

        # EMA 测试
        print("\n--- EMA 测试 ---")
        try:
            ema = EMA(model, decay=0.999)
            ema.update(); ema.apply_shadow(); ema.restore()
            print("EMA 创建/更新/切换/恢复 成功")
        except Exception as e:
            print(f"EMA 测试失败: {e}")

        # BN → GN 替换测试
        print("\n--- BN -> GN 替换测试 ---")
        try:
            model_gn = YFFC()
            bn_count = sum(1 for m in model_gn.modules() if isinstance(m, nn.BatchNorm2d))
            replace_bn_with_gn(model_gn)
            gn_count = sum(1 for m in model_gn.modules() if isinstance(m, nn.GroupNorm))
            bn_after = sum(1 for m in model_gn.modules() if isinstance(m, nn.BatchNorm2d))
            print(f"替换前 BN: {bn_count}, 替换后 GN: {gn_count}, 剩余 BN: {bn_after}")
            model_gn.to(device)
            with torch.no_grad():
                out_gn = model_gn(test_input)
            assert out_gn["r"].shape == test_input.shape
            assert out_gn["s"].shape == test_input.shape
            print("BN->GN 替换后前向传播正常")
            del model_gn
        except Exception as e:
            print(f"BN->GN 测试失败: {e}")

        if args.show_params:
            display_basic_model_info(model)

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)

    elif args.command == 'train':
        if hasattr(args, 'no_augmentation') and args.no_augmentation:
            args.use_augmentation = False
        if hasattr(args, 'no_ohem') and args.no_ohem:
            args.use_ohem = False

        image_size = (args.image_size, args.image_size)
        cfg = {
            'EPOCHS': args.epochs,
            'BATCH_SIZE': args.batch_size,
            'BASE_LR': args.base_lr,
            'WEIGHT_DECAY': args.weight_decay,
            'ACCUMULATION_STEPS': args.accumulation_steps,
            'GRAD_CLIP_VALUE': 1.0,
            'IMAGE_SIZE': image_size,
            'TRAIN_SPLIT_RATIO': args.train_split_ratio,
            'NUM_WORKERS': args.num_workers,
            'USE_GN': args.use_gn,
            'USE_EMA': args.use_ema,
            'EMA_DECAY': args.ema_decay,
            'USE_OHEM': args.use_ohem,
            'USE_AMP': False,
            'USE_HIGHLIGHT_AUG': args.use_augmentation,
            'HIGHLIGHT_AUG_P': 0.5,
            'INPUT_DIR': args.input_dir,
            'TARGET_DIR': args.target_dir,
            'VAL_INPUT_DIR': args.val_input_dir,
            'VAL_TARGET_DIR': args.val_target_dir,
            'SAVE_EVERY_N_EPOCHS': args.save_interval,
            'VIS_EVERY_N_EPOCHS': 10,
            'VIS_NUM_SAMPLES': 2,
            'VIS_SAMPLE_SEED': None,
            'USE_EARLY_STOPPING': False,
            'EARLY_STOPPING_PATIENCE': 30,
            'OUTPUT_FOLDER': args.save_dir,
        }
        train_model(cfg)
