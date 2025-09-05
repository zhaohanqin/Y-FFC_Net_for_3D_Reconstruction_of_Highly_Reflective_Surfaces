import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from torch.utils.data import DataLoader  # 数据加载器
import os  # 文件与路径操作
import numpy as np  # 数值计算
from PIL import Image  # 图像读写
import matplotlib.pyplot as plt  # 可视化
from matplotlib import rcParams  # 字体/渲染设置
from tqdm import tqdm  # 进度条
from Y_FFC import YFFC, FringeDataset, SSIM, statistical_loss  # 导入模型、数据集与损失

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda'):
    """
    训练模型：执行多轮（epoch）训练与验证，并保存最佳模型权重
    
    参数:
        model: Y-FFC网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数，默认100轮
        lr: 学习率，默认0.001
        device: 计算设备，默认CUDA
    
    返回:
        train_losses: 训练损失历史记录
        val_losses: 验证损失历史记录
    """
    model.to(device)  # 将模型移动到指定设备（CPU/GPU）
    
    # 初始化优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器，适合大多数深度学习任务
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # 学习率调度器：每30轮衰减为0.5
    
    # 定义损失函数
    ssim_loss = SSIM()  # 结构相似性损失（值越大越好，这里用 1-SSIM 转换为损失）
    mae_loss = nn.L1Loss()  # 平均绝对误差损失，用于像素级精度
    
    # 记录训练/验证历史损失，用于绘制学习曲线
    train_losses = []  # 每轮训练平均损失
    val_losses = []    # 每轮验证平均损失
    
    best_val_loss = float('inf')  # 最佳验证损失（越小越好），用于模型保存
    
    print(f"开始训练，共{epochs}轮，学习率{lr}")
    
    for epoch in range(epochs):  # 逐轮训练
        # ==================== 训练阶段 ====================
        model.train()  # 切换到训练模式（启用BN/Dropout等正则化）
        train_loss = 0  # 本轮训练损失累计
        
        # 使用 tqdm 展示训练批次进度
        for batch_idx, batch in enumerate(tqdm(train_loader, total=len(train_loader), desc=f'Train {epoch+1}/{epochs}', leave=False)):
            # 将数据移到指定设备
            inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]
            
            optimizer.zero_grad()  # 清空上一轮的梯度，防止梯度累积
            
            # ==================== 前向传播 ====================
            estimations = model(inputs)  # 模型输出字典：{'r': 漫反射, 'g': 梯度, 'b': 镜面反射}
            pred_diffuse = estimations['r']    # 预测漫反射分量
            pred_specular = estimations['b']   # 预测镜面反射分量
            pred_gradient = estimations['g']   # 预测梯度分量
            
            # ==================== 损失计算 ====================
            # 漫反射分支损失计算
            ld_ssim = 1 - ssim_loss(pred_diffuse, gt_diffuse)  # 结构相似性损失
            ld_mae = mae_loss(pred_diffuse, gt_diffuse)         # L1 损失
            ld_sl = statistical_loss(pred_diffuse, gt_diffuse)  # 统计损失（均值/方差一致性）
            ld = ld_ssim + ld_mae + ld_sl  # 漫反射总损失
            
            # 镜面反射分支损失计算
            ls_ssim = 1 - ssim_loss(pred_specular, gt_specular)  # 结构相似性损失
            ls_sl = statistical_loss(pred_specular, gt_specular)  # 统计损失
            ls = ls_ssim + ls_sl  # 镜面反射总损失
            
            # 梯度分支损失计算
            lg_ssim = 1 - ssim_loss(pred_gradient, gt_gradient)  # 结构相似性损失
            lg = lg_ssim  # 梯度损失（仅使用SSIM）
            
            # 总损失 = 漫反射损失 + 镜面反射损失 + 梯度损失
            loss = ld + ls + lg
            
            # ==================== 反向传播与参数更新 ====================
            loss.backward()  # 计算梯度
            optimizer.step()  # 按梯度更新权重（真正的"训练"发生在这里）
            
            train_loss += loss.item()  # 累计训练损失
            
            # 在进度条上显示当前损失
            if (batch_idx + 1) % 10 == 0:
                tqdm.write(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.6f}')
        
        # ==================== 验证阶段 ====================
        model.eval()  # 切换到评估模式（禁用BN/Dropout等）
        val_loss = 0  # 本轮验证损失累计
        
        with torch.no_grad():  # 关闭梯度计算以节省显存与加速
            # 使用 tqdm 展示验证批次进度
            for batch in tqdm(val_loader, total=len(val_loader), desc=f'Val {epoch+1}/{epochs}', leave=False):
                inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]
                
                # 前向传播（不计算梯度）
                estimations = model(inputs)
                pred_diffuse = estimations['r']
                pred_specular = estimations['b']
                pred_gradient = estimations['g']
                
                # 验证损失计算（与训练阶段相同）
                ld_ssim = 1 - ssim_loss(pred_diffuse, gt_diffuse)
                ld_mae = mae_loss(pred_diffuse, gt_diffuse)
                ld_sl = statistical_loss(pred_diffuse, gt_diffuse)
                ld = ld_ssim + ld_mae + ld_sl
                
                ls_ssim = 1 - ssim_loss(pred_specular, gt_specular)
                ls_sl = statistical_loss(pred_specular, gt_specular)
                ls = ls_ssim + ls_sl
                
                lg_ssim = 1 - ssim_loss(pred_gradient, gt_gradient)
                lg = lg_ssim
                
                loss = ld + ls + lg
                val_loss += loss.item()
        
        # ==================== 统计与记录 ====================
        # 计算本轮的平均训练/验证损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 记录损失历史
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # 保存最佳模型（以验证损失为准，防止过拟合）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'最佳模型已保存，验证损失: {best_val_loss:.6f}')
        
        scheduler.step()  # 学习率步进（每30轮衰减）
    
    print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")
    return train_losses, val_losses  # 返回训练过程中的损失曲线

def test_model(model, test_loader, device='cuda'):
    """
    测试模型：在测试集上计算评估指标（SSIM/MAE/PSNR），不更新参数
    
    参数:
        model: 训练好的Y-FFC网络模型
        test_loader: 测试数据加载器
        device: 计算设备
    
    返回:
        avg_ssim: 平均结构相似性指数
        avg_mae: 平均绝对误差
        avg_psnr: 平均峰值信噪比
    """
    model.eval()  # 切换到评估模式
    
    # 初始化损失函数（用于计算指标）
    ssim_loss = SSIM()  # 用于计算 SSIM 指标
    mae_loss = nn.L1Loss()  # 用于计算 MAE 指标
    
    # 累计指标统计
    total_ssim = 0  # SSIM 累计
    total_mae = 0   # MAE 累计
    total_psnr = 0  # PSNR 累计
    
    print("开始模型测试...")
    
    with torch.no_grad():  # 不计算梯度，节省内存
        for batch_idx, batch in enumerate(tqdm(test_loader, total=len(test_loader), desc='Testing', leave=True)):
            inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]
            
            # 前向传播
            estimations = model(inputs)
            pred_diffuse = estimations['r']     # 漫反射预测
            pred_specular = estimations['b']    # 镜面反射预测（这里未直接用于指标）
            pred_gradient = estimations['g']    # 梯度预测（这里未直接用于指标）
            
            # ==================== 计算评估指标 ====================
            ssim_val = ssim_loss(pred_diffuse, gt_diffuse)  # 结构相似性
            mae_val = mae_loss(pred_diffuse, gt_diffuse)    # 平均绝对误差
            
            # PSNR 计算：基于均方误差，值越大表示图像质量越好
            mse = nn.MSELoss()(pred_diffuse, gt_diffuse)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # PSNR = 20*log10(MAX/√MSE)
            
            # 累计指标
            total_ssim += ssim_val.item()
            total_mae += mae_val.item()
            total_psnr += psnr.item()
            
            # 在进度条上输出每个 batch 的指标
            tqdm.write(f'Batch {batch_idx}: SSIM: {ssim_val:.4f}, MAE: {mae_val:.4f}, PSNR: {psnr:.2f}')
    
    # ==================== 计算平均指标 ====================
    avg_ssim = total_ssim / len(test_loader)
    avg_mae = total_mae / len(test_loader)
    avg_psnr = total_psnr / len(test_loader)
    
    print(f'\n========== 测试结果汇总 ==========')
    print(f'SSIM: {avg_ssim:.4f} (越接近1越好)')
    print(f'MAE:  {avg_mae:.4f} (越接近0越好)')
    print(f'PSNR: {avg_psnr:.2f} dB (越大越好)')
    
    return avg_ssim, avg_mae, avg_psnr  # 返回测试集上的平均指标

def visualize_results(model, test_loader, device='cuda', num_samples=5):
    """
    可视化测试结果：显示输入、预测与差异图，帮助理解模型效果
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        num_samples: 要可视化的样本数量
    """
    model.eval()  # 评估模式
    
    print(f"开始可视化{num_samples}个测试样本...")
    
    with torch.no_grad():  # 不进行梯度计算
        for i, batch in enumerate(tqdm(test_loader, total=min(len(test_loader), num_samples), desc='Visualizing', leave=True)):
            if i >= num_samples:  # 只可视化指定数量样本
                break
                
            inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]
            
            # 前向传播获取预测结果
            estimations = model(inputs)
            pred_diffuse = estimations['r']     # 预测漫反射
            pred_specular = estimations['b']    # 预测镜面反射
            pred_gradient = estimations['g']    # 预测梯度
            
            # ==================== 数据转换 ====================
            # 转为 numpy 数组便于显示（取第一个样本）
            input_img = inputs[0, 0].cpu().numpy()           # 输入图像
            gt_diffuse_img = gt_diffuse[0, 0].cpu().numpy() # 真实漫反射
            pred_diffuse_img = pred_diffuse[0, 0].cpu().numpy() # 预测漫反射
            pred_specular_img = pred_specular[0, 0].cpu().numpy() # 预测镜面反射
            pred_gradient_img = pred_gradient[0, 0].cpu().numpy() # 预测梯度
            
            # ==================== 创建可视化布局 ====================
            # 2×3 网格布局，显示所有相关图像
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 第一行：输入、真实漫反射、预测漫反射
            axes[0, 0].imshow(input_img, cmap='gray')
            # 英文标题释义：Input (Highly Reflective) = “输入图像（高反射表面）”
            axes[0, 0].set_title('Input (Highly Reflective)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(gt_diffuse_img, cmap='gray')
            # 英文标题释义：Ground Truth (Diffuse) = “真实标签（漫反射分量）”
            axes[0, 1].set_title('Ground Truth (Diffuse)')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(pred_diffuse_img, cmap='gray')
            # 英文标题释义：Predicted (Diffuse) = “网络预测（漫反射分量）”
            axes[0, 2].set_title('Predicted (Diffuse)')
            axes[0, 2].axis('off')
            
            # 第二行：预测镜面反射、预测梯度、差异图
            axes[1, 0].imshow(pred_specular_img, cmap='gray')
            # 英文标题释义：Predicted (Specular) = “网络预测（镜面反射分量）”
            axes[1, 0].set_title('Predicted (Specular)')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(pred_gradient_img, cmap='gray')
            # 英文标题释义：Predicted (Gradient) = “网络预测（梯度场）”
            axes[1, 1].set_title('Predicted (Gradient)')
            axes[1, 1].axis('off')
            
            # 显示与真实漫反射的差异（用于评估预测精度）
            diff = np.abs(pred_diffuse_img - gt_diffuse_img)
            axes[1, 2].imshow(diff, cmap='hot')  # 热力图显示差异
            # 英文标题释义：Difference (|Pred - GT|) = “预测与真实的绝对差异”
            axes[1, 2].set_title('Difference (|Pred - GT|)')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'results_sample_{i}.png', dpi=300, bbox_inches='tight')  # 保存高分辨率结果
            plt.show()
            
            print(f"样本 {i+1} 可视化完成，已保存为 results_sample_{i}.png")

def create_dummy_data(input_dir, target_dir, num_samples=100):
    """
    创建虚拟数据集：生成带反射的输入与无反射的目标，用于快速测试训练流程
    
    参数:
        input_dir: 输入图像保存目录
        target_dir: 目标图像保存目录
        num_samples: 生成的样本数量
    """
    os.makedirs(input_dir, exist_ok=True)  # 创建输入目录
    os.makedirs(target_dir, exist_ok=True)  # 创建目标目录
    
    print(f"开始生成{num_samples}个虚拟样本...")
    
    for i in range(num_samples):  # 生成指定数量的样本
        # ==================== 生成基础条纹图案 ====================
        size = 512  # 图像尺寸
        x, y = np.meshgrid(np.linspace(0, 2*np.pi, size), np.linspace(0, 2*np.pi, size))  # 创建坐标网格
        
        # 基础正弦条纹图案（模拟真实条纹投影）
        base_pattern = 0.5 + 0.3 * np.sin(10*x + 5*y)  # 基础正弦条纹
        noise = np.random.normal(0, 0.05, (size, size))  # 添加高斯噪声（模拟真实环境）
        
        # 稀疏反射效果（模拟高反射表面的镜面反射）
        reflection = np.random.uniform(0, 0.4, (size, size)) * (np.random.random((size, size)) > 0.8)
        
        # ==================== 生成输入和目标图像 ====================
        input_pattern = np.clip(base_pattern + noise + reflection, 0, 1)  # 高反射输入（包含镜面反射）
        target_pattern = np.clip(base_pattern + noise, 0, 1)              # 漫反射目标（无镜面反射）
        
        # 保存图像
        input_img = Image.fromarray((input_pattern * 255).astype(np.uint8))
        target_img = Image.fromarray((target_pattern * 255).astype(np.uint8))
        input_img.save(os.path.join(input_dir, f'input_{i:04d}.png'))
        target_img.save(os.path.join(target_dir, f'target_{i:04d}.png'))
    
    print(f"虚拟数据集生成完成！共{num_samples}个样本")
    print(f"输入图像保存在: {input_dir}")
    print(f"目标图像保存在: {target_dir}")

def main():
    """
    主函数：执行完整的训练、测试和可视化流程
    """
    print("=" * 60)
    print("Y-FFC Net 训练程序启动")
    print("=" * 60)
    
    # ==================== 设备设置 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # ==================== 数据路径设置 ====================
    input_dir = 'data/inputs'   # 输入图像目录
    target_dir = 'data/targets' # 目标图像目录
    
    # 如果数据目录不存在，自动生成虚拟数据用于测试
    if not os.path.exists(input_dir) or not os.path.exists(target_dir):
        print("检测到数据目录不存在，正在生成虚拟数据集...")
        create_dummy_data(input_dir, target_dir, num_samples=200)
    
    # ==================== 数据集构建与划分 ====================
    print("构建数据集...")
    train_dataset = FringeDataset(input_dir, target_dir)
    train_size = int(0.8 * len(train_dataset))  # 8:2 训练验证划分
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    print(f"数据集大小: 训练集 {train_size}, 验证集 {val_size}")
    
    # ==================== 数据加载器创建 ====================
    # Windows 下建议使用 num_workers=0 避免多进程问题
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # ==================== 数据维度检查 ====================
    print("检查数据维度...")
    try:
        sample_batch = next(iter(train_loader))
        print('样本批次形状:', [tuple(x.shape) for x in sample_batch])
        print('数据格式: [batch_size, channels, height, width]')
    except Exception as e:
        print('获取样本批次失败:', e)
        return
    
    # ==================== 模型创建 ====================
    print("创建Y-FFC网络模型...")
    model = YFFC()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # ==================== 模型训练 ====================
    print("\n" + "=" * 40)
    print("开始模型训练...")
    print("=" * 40)
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, device=device)
    
    # ==================== 模型测试 ====================
    print("\n" + "=" * 40)
    print("开始模型测试...")
    print("=" * 40)
    # 加载训练过程中保存的最佳模型权重
    model.load_state_dict(torch.load('best_model.pth'))
    test_model(model, test_loader, device=device)
    
    # ==================== 结果可视化 ====================
    print("\n" + "=" * 40)
    print("开始结果可视化...")
    print("=" * 40)
    visualize_results(model, test_loader, device=device, num_samples=3)
    
    # ==================== 训练历史可视化 ====================
    print("保存训练历史曲线...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失', linewidth=2)
    plt.plot(val_losses, label='验证损失', linewidth=2)
    plt.xlabel('训练轮数 (Epoch)')
    plt.ylabel('损失值 (Loss)')
    plt.title('Y-FFC Net 训练历史')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("所有流程完成！")
    print("=" * 60)

if __name__ == '__main__':  # 脚本入口
    # ========== Matplotlib 字体设置 ==========
    # 解决中文显示为乱码以及坐标轴负号显示问题
    # Windows 常见可用中文字体：Microsoft YaHei / SimHei
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
    rcParams['axes.unicode_minus'] = False
    main()  # 运行主流程