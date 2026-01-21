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
import time  # 时间跟踪
from datetime import datetime, timedelta  # 日期时间处理
from Y_FFC import YFFC, FringeDataset, SSIM, statistical_loss  # 导入模型、数据集与损失

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda', 
                use_amp=True, gradient_accumulation_steps=1, optimizer_config=None, 
                scheduler_config=None, loss_config=None, training_config=None):
    """
    训练模型：执行多轮（epoch）训练与验证，并保存最佳模型权重
    
    参数:
        model: Y-FFC网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数，默认100轮
        lr: 学习率，默认0.001
        device: 计算设备，默认CUDA
        use_amp: 是否使用混合精度训练，默认True（节省显存）
        gradient_accumulation_steps: 梯度累积步数，默认1（增大此值可模拟更大的batch_size）
        optimizer_config: 优化器配置字典
        scheduler_config: 学习率调度器配置字典
        loss_config: 损失函数配置字典
        training_config: 训练策略配置字典
    
    返回:
        train_losses: 训练损失历史记录
        val_losses: 验证损失历史记录
    """
    # 设置默认配置
    if optimizer_config is None:
        optimizer_config = {'type': 'adam', 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0}
    if scheduler_config is None:
        scheduler_config = {'type': 'step', 'step_size': 30, 'gamma': 0.5}
    if loss_config is None:
        loss_config = {'ssim_window_size': 11, 'ssim_sigma': 1.5, 
                      'weight_diffuse': 1.0, 'weight_specular': 1.0, 'weight_gradient': 1.0}
    if training_config is None:
        # 默认梯度判别模块启用epoch = 总训练轮数的40%
        default_enable_gd_epoch = int(epochs * 0.4)
        training_config = {'use_grad_clip': False, 'grad_clip_value': 1.0, 
                          'use_early_stopping': False, 'early_stopping_patience': 20,
                          'model_save_path': 'best_model.pth', 'checkpoint_dir': 'checkpoints',
                          'save_every_n_epochs': 0,
                          'enable_gradient_discriminator_epoch': default_enable_gd_epoch,  # 从第几个epoch开始启用梯度判别模块，默认值为总训练轮数的40%
                          'gradient_constraint_weight': 0.3}  # 梯度约束权重（弱约束），范围[0,1]
    # 清理CUDA缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"已清理CUDA缓存，当前显存使用: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
    
    model.to(device)  # 将模型移动到指定设备（CPU/GPU）
    
    # ==================== 初始化优化器 ====================
    opt_type = optimizer_config.get('type', 'adam').lower()
    if opt_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=lr,
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
        print(f"使用Adam优化器: lr={lr}, betas={optimizer_config.get('betas', (0.9, 0.999))}, weight_decay={optimizer_config.get('weight_decay', 0.0)}")
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=optimizer_config.get('weight_decay', 0.01)
        )
        print(f"使用AdamW优化器: lr={lr}, weight_decay={optimizer_config.get('weight_decay', 0.01)}")
    elif opt_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 0.0001)
        )
        print(f"使用SGD优化器: lr={lr}, momentum={optimizer_config.get('momentum', 0.9)}")
    else:
        raise ValueError(f"不支持的优化器类型: {opt_type}")
    
    # ==================== 初始化学习率调度器 ====================
    sched_type = scheduler_config.get('type', 'step').lower()
    if sched_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.5)
        )
        print(f"使用StepLR调度器: step_size={scheduler_config.get('step_size', 30)}, gamma={scheduler_config.get('gamma', 0.5)}")
    elif sched_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', epochs),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
        print(f"使用CosineAnnealingLR调度器: T_max={scheduler_config.get('T_max', epochs)}")
    elif sched_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 10),
            verbose=True
        )
        print(f"使用ReduceLROnPlateau调度器: patience={scheduler_config.get('patience', 10)}")
    elif sched_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.95)
        )
        print(f"使用ExponentialLR调度器: gamma={scheduler_config.get('gamma', 0.95)}")
    else:
        scheduler = None
        print("未使用学习率调度器")
    
    # 混合精度训练scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    
    # ==================== 定义损失函数 ====================
    ssim_window_size = loss_config.get('ssim_window_size', 11)
    ssim_sigma = loss_config.get('ssim_sigma', 1.5)
    ssim_loss = SSIM(window_size=ssim_window_size, sigma=ssim_sigma)
    mae_loss = nn.L1Loss()
    print(f"SSIM损失参数: window_size={ssim_window_size}, sigma={ssim_sigma}")
    
    # 损失权重
    weight_diffuse = loss_config.get('weight_diffuse', 1.0)
    weight_specular = loss_config.get('weight_specular', 1.0)
    weight_gradient = loss_config.get('weight_gradient', 1.0)
    print(f"损失权重: 漫反射={weight_diffuse}, 镜面反射={weight_specular}, 梯度={weight_gradient}")
    
    # 记录训练/验证历史损失，用于绘制学习曲线
    train_losses = []  # 每轮训练平均损失
    val_losses = []    # 每轮验证平均损失
    train_accuracies = []  # 每轮训练平均SSIM值
    val_accuracies = []    # 每轮验证平均SSIM值
    epoch_times = []   # 每轮训练时间
    
    best_val_loss = float('inf')  # 最佳验证损失（越小越好），用于模型保存
    
    # ==================== 训练策略配置 ====================
    use_grad_clip = training_config.get('use_grad_clip', False)
    grad_clip_value = training_config.get('grad_clip_value', 1.0)
    use_early_stopping = training_config.get('use_early_stopping', False)
    early_stopping_patience = training_config.get('early_stopping_patience', 20)
    early_stopping_counter = 0
    model_save_path = training_config.get('model_save_path', 'best_model.pth')
    checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
    save_every_n_epochs = training_config.get('save_every_n_epochs', 0)
    
    # 创建检查点目录
    if save_every_n_epochs > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    if use_grad_clip:
        print(f"启用梯度裁剪: max_norm={grad_clip_value}")
    if use_early_stopping:
        print(f"启用早停策略: patience={early_stopping_patience}")
    
    print(f"开始训练，共{epochs}轮，学习率{lr}")
    if use_amp and device.type == 'cuda':
        print("使用混合精度训练（AMP）以节省显存")
    if gradient_accumulation_steps > 1:
        print(f"使用梯度累积，累积步数: {gradient_accumulation_steps}")
    
    # 打印训练开始时间
    start_time_total = time.time()
    print(f"\n 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # ==================== 梯度判别模块延后启用配置 ====================
    enable_gd_epoch = training_config.get('enable_gradient_discriminator_epoch', 0)
    gradient_constraint_weight = training_config.get('gradient_constraint_weight', 0.3)
    
    # 初始化梯度判别模块状态
    if hasattr(model, 'enable_gradient_discriminator'):
        model.enable_gradient_discriminator = (enable_gd_epoch == 0)
        model.gradient_constraint_weight = gradient_constraint_weight
        if enable_gd_epoch > 0:
            print(f"\n梯度判别模块将在第 {enable_gd_epoch} 个epoch启用")
            print(f"梯度约束权重: {gradient_constraint_weight} (弱约束)")
    
    for epoch in range(epochs):  # 逐轮训练
        epoch_start_time = time.time()  # 记录本轮开始时间
        
        # ==================== 动态启用梯度判别模块 ====================
        if hasattr(model, 'enable_gradient_discriminator'):
            # 根据epoch数决定是否启用梯度判别模块
            should_enable = (epoch >= enable_gd_epoch)
            if model.enable_gradient_discriminator != should_enable:
                model.enable_gradient_discriminator = should_enable
                if should_enable:
                    print(f"\n[Epoch {epoch+1}] 梯度判别模块已启用，约束权重: {gradient_constraint_weight}")
                else:
                    print(f"\n[Epoch {epoch+1}] 梯度判别模块已禁用（将在第 {enable_gd_epoch} 个epoch启用）")
        
        # ==================== 训练阶段 ====================
        model.train()  # 切换到训练模式（启用BN/Dropout等正则化）
        train_loss_sum = 0.0  # 本轮训练损失（按样本加权）累计
        train_ssim_sum = 0.0  # 本轮训练SSIM（按样本加权）累计
        train_num_samples = 0  # 本轮训练样本数累计
        
        # 使用 tqdm 展示训练批次进度
        optimizer.zero_grad()  # 在循环开始前清空梯度
        
        for batch_idx, batch in enumerate(tqdm(train_loader, total=len(train_loader), desc=f'Train {epoch+1}/{epochs}', leave=False)):
            # 将数据移到指定设备
            inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]
            batch_size = inputs.size(0)
            
            # ==================== 前向传播（使用混合精度） ====================
            if use_amp and scaler is not None:
                with torch.amp.autocast('cuda'):
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
                    
                    # 总损失 = 漫反射损失 + 镜面反射损失 + 梯度损失（加权）
                    loss = weight_diffuse * ld + weight_specular * ls + weight_gradient * lg
                    loss = loss / gradient_accumulation_steps  # 梯度累积时缩放损失
                    
                    # 计算SSIM值（用于训练统计，按样本加权）
                    train_ssim_sum += ssim_loss(pred_diffuse, gt_diffuse).item() * batch_size
            else:
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
                
                # 总损失 = 漫反射损失 + 镜面反射损失 + 梯度损失（加权）
                loss = weight_diffuse * ld + weight_specular * ls + weight_gradient * lg
                loss = loss / gradient_accumulation_steps  # 梯度累积时缩放损失
                
                # 计算SSIM值（用于训练统计，按样本加权）
                train_ssim_sum += ssim_loss(pred_diffuse, gt_diffuse).item() * batch_size
            
            # ==================== 反向传播与参数更新 ====================
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()  # 混合精度反向传播
            else:
                loss.backward()  # 计算梯度
            
            # 梯度累积：每accumulation_steps步更新一次参数
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪（防止梯度爆炸）
                if use_grad_clip:
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                
                if use_amp and scaler is not None:
                    scaler.step(optimizer)  # 混合精度参数更新
                    scaler.update()  # 更新scaler
                else:
                    optimizer.step()  # 按梯度更新权重
                optimizer.zero_grad()  # 清空梯度
            
            # 累计训练损失（恢复原始尺度；按样本加权，避免最后一个小batch造成统计抖动）
            train_loss_sum += (loss.item() * gradient_accumulation_steps) * batch_size
            train_num_samples += batch_size
            
            # 在进度条上显示当前损失
            if (batch_idx + 1) % 10 == 0:
                tqdm.write(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item() * gradient_accumulation_steps:.6f}')
        
        # 处理最后一个不完整的累积批次
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # ==================== 验证阶段 ====================
        model.eval()  # 切换到评估模式（禁用BN/Dropout等）
        val_loss_sum = 0.0  # 本轮验证损失（按样本加权）累计
        val_ssim_sum = 0.0  # 本轮验证SSIM（按样本加权）累计
        val_num_samples = 0  # 本轮验证样本数累计
        
        with torch.no_grad():  # 关闭梯度计算以节省显存与加速
            # 使用 tqdm 展示验证批次进度
            for batch in tqdm(val_loader, total=len(val_loader), desc=f'Val {epoch+1}/{epochs}', leave=False):
                inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]
                batch_size = inputs.size(0)
                
                # 前向传播（不计算梯度，使用混合精度）
                if use_amp and scaler is not None:
                    with torch.amp.autocast('cuda'):
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
                        
                        loss = weight_diffuse * ld + weight_specular * ls + weight_gradient * lg
                        
                        # 计算SSIM值（用于验证统计，按样本加权）
                        val_ssim_sum += ssim_loss(pred_diffuse, gt_diffuse).item() * batch_size
                else:
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
                    
                    loss = weight_diffuse * ld + weight_specular * ls + weight_gradient * lg
                    
                    # 计算SSIM值（用于验证统计，按样本加权）
                    val_ssim_sum += ssim_loss(pred_diffuse, gt_diffuse).item() * batch_size
                
                # 按样本加权累计，避免最后一个小batch（如batch_size=1）被“等权重”放大影响
                val_loss_sum += loss.item() * batch_size
                val_num_samples += batch_size
        
        # ==================== 统计与记录 ====================
        # 计算本轮的平均训练/验证损失和SSIM值
        # 用“按样本数加权”的平均，验证曲线会显著更稳（尤其是验证集不能整除batch_size时）
        avg_train_loss = train_loss_sum / max(train_num_samples, 1)
        avg_val_loss = val_loss_sum / max(val_num_samples, 1)
        avg_train_acc = train_ssim_sum / max(train_num_samples, 1)  # 平均SSIM值
        avg_val_acc = val_ssim_sum / max(val_num_samples, 1)
        
        # 记录损失和SSIM历史
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(avg_train_acc)
        val_accuracies.append(avg_val_acc)
        
        # 计算本轮耗时和预估剩余时间
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta = timedelta(seconds=int(eta_seconds))
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # ==================== 打印详细训练信息 ====================
        print("\n" + "=" * 80)
        print(f"Epoch [{epoch+1}/{epochs}] 训练统计")
        print("=" * 80)
        print(f"本轮耗时: {epoch_time:.2f}秒 | 平均耗时: {avg_epoch_time:.2f}秒/轮 | 预计剩余: {eta}")
        print(f"损失值:   训练损失 = {avg_train_loss:.6f} | 验证损失 = {avg_val_loss:.6f}")
        print(f"SSIM值:   训练SSIM = {avg_train_acc:.4f} | 验证SSIM = {avg_val_acc:.4f}")
        print(f"超参数:   学习率 = {current_lr:.6f} | 批大小 = {train_loader.batch_size} | 累积步数 = {gradient_accumulation_steps}")
        if device.type == 'cuda':
            print(f"显存使用: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB / {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB (当前/峰值)")
        
        # ==================== 保存最佳模型 ====================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0  # 重置早停计数器
            torch.save(model.state_dict(), model_save_path)
            print(f'最佳模型已保存: {model_save_path}, 验证损失: {best_val_loss:.6f}, 验证SSIM: {avg_val_acc:.4f}')
        else:
            early_stopping_counter += 1
            if use_early_stopping:
                print(f'⚠️ 验证损失未改善 ({early_stopping_counter}/{early_stopping_patience})')
        
        # ==================== 实时绘制训练曲线 ====================
        if (epoch + 1) % 5 == 0 or epoch == 0:  # 每5轮或第一轮绘制一次
            output_folder = os.path.dirname(model_save_path)
            plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, 
                               output_folder, epoch + 1)
            print(f'训练曲线已更新: {os.path.join(output_folder, "training_curves_realtime.png")}')
        
        # ==================== 定期可视化样本结果 ====================
        if (epoch + 1) % 10 == 0 or epoch == 0:  # 每10轮或第一轮可视化一次
            output_folder = os.path.dirname(model_save_path)
            visualize_sample_during_training(model, val_loader, device, output_folder, epoch + 1)
            print(f'样本可视化已更新: {os.path.join(output_folder, f"sample_epoch_{epoch+1}.png")}')
        
        # ==================== 定期保存检查点 ====================
        if save_every_n_epochs > 0 and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'train_acc': avg_train_acc,
                'val_acc': avg_val_acc
            }, checkpoint_path)
            print(f'检查点已保存: {checkpoint_path}')
            print("=" * 80 + "\n")
        
        # ==================== 早停检查 ====================
        if use_early_stopping and early_stopping_counter >= early_stopping_patience:
            print(f'\n⛔早停触发！验证损失在 {early_stopping_patience} 轮内未改善，停止训练。')
            break
        
        # ==================== 学习率调度 ====================
        if scheduler is not None:
            if sched_type == 'plateau':
                scheduler.step(avg_val_loss)  # ReduceLROnPlateau需要验证损失
            else:
                scheduler.step()  # 其他调度器按epoch步进
    
    # 打印训练总结
    total_time = time.time() - start_time_total
    total_time_str = str(timedelta(seconds=int(total_time)))
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"总训练时间: {total_time_str}")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最终训练SSIM: {train_accuracies[-1]:.4f} | 最终验证SSIM: {val_accuracies[-1]:.4f}")
    print("=" * 80)
    
    return train_losses, val_losses, train_accuracies, val_accuracies  # 返回训练过程中的损失和SSIM曲线

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

def visualize_results(model, test_loader, device='cuda', num_samples=5, output_folder=''):
    """
    可视化测试结果：显示输入、预测与差异图，帮助理解模型效果
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        num_samples: 要可视化的样本数量
        output_folder: 输出文件夹路径
    """
    model.eval()  # 评估模式
    
    # 创建可视化结果子文件夹
    if output_folder:
        vis_folder = os.path.join(output_folder, 'visualizations')
        os.makedirs(vis_folder, exist_ok=True)
    else:
        vis_folder = ''
    
    # 处理num_samples为None的情况：可视化所有测试样本
    if num_samples is None:
        num_samples = len(test_loader)
    
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
            
            # 显示与真实漫反射的差异（用于评估预测质量）
            diff = np.abs(pred_diffuse_img - gt_diffuse_img)
            axes[1, 2].imshow(diff, cmap='hot')  # 热力图显示差异
            # 英文标题释义：Difference (|Pred - GT|) = “预测与真实的绝对差异”
            axes[1, 2].set_title('Difference (|Pred - GT|)')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # 保存到输出文件夹
            if vis_folder:
                save_path = os.path.join(vis_folder, f'results_sample_{i+1}.png')
            else:
                save_path = f'results_sample_{i+1}.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存高分辨率结果
            plt.close()  # 关闭图形，节省内存
            
            print(f"样本 {i+1} 可视化完成，已保存为 {save_path}")

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_folder, current_epoch):
    """
    实时绘制训练曲线（损失和SSIM）
    
    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练SSIM列表
        val_accs: 验证SSIM列表
        output_folder: 输出文件夹
        current_epoch: 当前epoch数
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-o', label='训练损失', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='验证损失', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失值 (Loss)', fontsize=12)
    ax1.set_title(f'训练和验证损失 (Epoch {current_epoch})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 绘制SSIM曲线
    ax2.plot(epochs, train_accs, 'b-o', label='训练SSIM', linewidth=2, markersize=4)
    ax2.plot(epochs, val_accs, 'r-s', label='验证SSIM', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('SSIM值', fontsize=12)
    ax2.set_title(f'训练和验证SSIM (Epoch {current_epoch})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])  # SSIM范围为0-1
    
    plt.tight_layout()
    save_path = os.path.join(output_folder, 'training_curves_realtime.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_sample_during_training(model, val_loader, device, output_folder, current_epoch):
    """
    训练过程中可视化样本结果
    
    参数:
        model: 模型
        val_loader: 验证数据加载器
        device: 计算设备
        output_folder: 输出文件夹
        current_epoch: 当前epoch数
    """
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        batch = next(iter(val_loader))
        inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]
        
        # 前向传播
        estimations = model(inputs)
        pred_diffuse = estimations['r']
        pred_specular = estimations['b']
        pred_gradient = estimations['g']
        
        # 转换为numpy
        input_img = inputs[0, 0].cpu().numpy()
        gt_diffuse_img = gt_diffuse[0, 0].cpu().numpy()
        pred_diffuse_img = pred_diffuse[0, 0].cpu().numpy()
        pred_specular_img = pred_specular[0, 0].cpu().numpy()
        pred_gradient_img = pred_gradient[0, 0].cpu().numpy()
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('输入图像', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt_diffuse_img, cmap='gray')
        axes[0, 1].set_title('真实漫反射', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred_diffuse_img, cmap='gray')
        axes[0, 2].set_title('预测漫反射', fontsize=12)
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(pred_specular_img, cmap='gray')
        axes[1, 0].set_title('预测镜面反射', fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred_gradient_img, cmap='gray')
        axes[1, 1].set_title('预测梯度', fontsize=12)
        axes[1, 1].axis('off')
        
        diff = np.abs(pred_diffuse_img - gt_diffuse_img)
        im = axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('差异图', fontsize=12)
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
        
        fig.suptitle(f'Epoch {current_epoch} - 样本可视化', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_folder, f'sample_epoch_{current_epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    model.train()  # 恢复训练模式

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
    
    # ==================== 训练参数配置区域 ====================
    # 在这里修改训练参数，无需深入代码内部
    
    # ========== 基础训练参数 ==========
    # 训练轮数
    EPOCHS = 200  # 训练轮数，建议范围：50-200
    
    # 学习率
    LEARNING_RATE = 0.0001  # 初始学习率，建议范围：0.0001-0.01
    
    # 批次大小
    BATCH_SIZE = 20  # 每批处理的样本数，显存不足时使用1，充足时可用2-4
    
    # 梯度累积步数（用于模拟更大的batch_size）
    GRADIENT_ACCUMULATION_STEPS = 3  # 梯度累积步数，等效batch_size = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS
    
    # ========== 优化器参数 ==========
    # 优化器类型：'adam', 'sgd', 'adamw'
    OPTIMIZER_TYPE = 'adam'  # Adam: 自适应学习率，适合大多数任务
    
    # Adam优化器参数
    ADAM_BETAS = (0.9, 0.999)  # Adam的beta参数，控制动量和二阶矩估计
    ADAM_EPS = 1e-8  # Adam的epsilon，防止除零
    ADAM_WEIGHT_DECAY = 0.01  # L2正则化系数，0表示不使用，建议范围：0-0.01
    
    # SGD优化器参数（仅当OPTIMIZER_TYPE='sgd'时使用）
    SGD_MOMENTUM = 0.94  # SGD动量，建议范围：0.9-0.99
    SGD_WEIGHT_DECAY = 0.0001  # SGD权重衰减
    
    # ========== 学习率调度器参数 ==========
    # 学习率调度器类型：'step', 'cosine', 'plateau', 'exponential'
    SCHEDULER_TYPE = 'step'  # step: 阶梯式衰减，cosine: 余弦退火，plateau: 自适应衰减
    
    # StepLR参数（阶梯式衰减）
    STEP_SIZE = 30  # 每隔多少轮衰减一次学习率
    GAMMA = 0.5  # 学习率衰减系数，新学习率 = 旧学习率 × gamma
    
    # CosineAnnealingLR参数（余弦退火）
    T_MAX = 50  # 余弦周期，通常设置为总训练轮数
    ETA_MIN = 1e-6  # 最小学习率
    
    # ReduceLROnPlateau参数（自适应衰减）
    PATIENCE = 10  # 验证损失不下降的容忍轮数
    PLATEAU_FACTOR = 0.5  # 衰减系数
    
    # ========== 损失函数参数 ==========
    # SSIM损失函数参数
    SSIM_WINDOW_SIZE = 11  # SSIM高斯窗口大小，建议范围：7-15（奇数）
    SSIM_SIGMA = 1.5  # SSIM高斯标准差，建议范围：1.0-2.0
    
    # 损失权重配置
    LOSS_WEIGHT_DIFFUSE = 1.0  # 漫反射损失权重
    LOSS_WEIGHT_SPECULAR = 1.0  # 镜面反射损失权重
    LOSS_WEIGHT_GRADIENT = 1.0  # 梯度损失权重
    
    # ========== 图像和数据参数 ==========
    # 图像尺寸设置
    # 选项1：正方形图像 - 设置为单个整数，例如：IMAGE_RESIZE = 1024
    # 选项2：矩形图像 - 设置为元组 (高度, 宽度)，例如：IMAGE_RESIZE = (768, 1024)
    # 选项3：使用原始尺寸 - 设置为 None
    # 注意：图像尺寸越大，显存占用越多。建议从512开始，根据显存情况逐步增大
    IMAGE_RESIZE = (160,128)  # 推荐尺寸：512(4GB显存), 768(8GB显存), 1024(12GB+显存)
    
    # 数据集划分比例
    TRAIN_SPLIT_RATIO = 0.9  # 训练集比例，0.8表示80%训练，20%验证
    
    # 数据加载器参数
    NUM_WORKERS = 0  # 数据加载工作进程数，Windows下建议使用0，Linux可以设置为2-4
    PIN_MEMORY = True  # 是否将数据固定在内存中（加速GPU传输）
    SHUFFLE_TRAIN = True  # 是否打乱训练数据
    
    # ========== 训练策略参数 ==========
    # 混合精度训练（节省显存）
    USE_AMP = False  # True=启用混合精度训练，False=使用全精度
    
    # 梯度裁剪（防止梯度爆炸）
    USE_GRAD_CLIP = False  # 是否启用梯度裁剪
    GRAD_CLIP_VALUE = 1.0  # 梯度裁剪阈值
    
    # 早停策略
    USE_EARLY_STOPPING = False  # 是否启用早停
    EARLY_STOPPING_PATIENCE = 20  # 验证损失不下降的最大容忍轮数
    
    # ========== 梯度判别模块参数 ==========
    # 延后启用梯度判别模块（缓解过拟合）
    # 默认值为总训练轮数的40%，如果设置为0表示从一开始就启用
    ENABLE_GRADIENT_DISCRIMINATOR_EPOCH = None  # None表示使用默认值（总训练轮数的40%），也可以手动指定具体数值
    GRADIENT_CONSTRAINT_WEIGHT = 0.3  # 梯度约束权重（弱约束），范围[0,1]，默认0.3
    # 较小的权重可以降低模型的决策敏感度，缓解验证集损失波动问题
    
    # ==================== 模型保存和可视化 ==========
    # 创建带时间戳的实验文件夹（与train_enhanced.py保持一致）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FOLDER = f'experiments/original_yffc_{timestamp}'  # 主输出文件夹
    
    # 创建输出目录结构
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/models", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/logs", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/visualizations", exist_ok=True)
    
    print(f"\n 实验输出目录: {os.path.abspath(OUTPUT_FOLDER)}")
    print("   所有训练结果将保存在此文件夹中\n")
    
    # 模型保存路径（相对于OUTPUT_FOLDER）
    MODEL_SAVE_PATH = 'best_model.pth'  # 最佳模型保存路径
    CHECKPOINT_DIR = 'checkpoints'  # 检查点保存目录
    SAVE_EVERY_N_EPOCHS = 5  # 每隔N轮保存一次检查点（0表示不保存）
    
    # 可视化样本数量
    VISUALIZATION_SAMPLES = None  # 测试后可视化的样本数量
    
    # 虚拟数据生成数量（仅在数据目录不存在时使用）
    # DUMMY_DATA_SAMPLES = 200  # 生成的虚拟样本数量
    
    # 更新路径为绝对路径
    MODEL_SAVE_PATH = os.path.join(OUTPUT_FOLDER, 'models', MODEL_SAVE_PATH)
    CHECKPOINT_DIR = os.path.join(OUTPUT_FOLDER, 'checkpoints')
    
    print("\n当前训练配置:")
    print(f"  - 训练轮数: {EPOCHS}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 梯度累积步数: {GRADIENT_ACCUMULATION_STEPS} (等效batch_size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  - 图像尺寸: {IMAGE_RESIZE if IMAGE_RESIZE else '原始尺寸'}")
    print(f"  - 训练集比例: {TRAIN_SPLIT_RATIO}")
    print(f"  - 输出文件夹: {OUTPUT_FOLDER}")
    print(f"  - 混合精度训练: {'启用' if USE_AMP else '禁用'}")
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
        create_dummy_data(input_dir, target_dir, num_samples=500)
    
    # ==================== 数据集构建与划分 ====================
    print("构建数据集...")
    train_dataset = FringeDataset(input_dir, target_dir, resize=IMAGE_RESIZE)
    train_size = int(TRAIN_SPLIT_RATIO * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    print(f"数据集大小: 训练集 {train_size}, 验证集 {val_size}")
    if IMAGE_RESIZE:
        print(f"图像已resize到: {IMAGE_RESIZE if isinstance(IMAGE_RESIZE, int) else f'{IMAGE_RESIZE[0]}x{IMAGE_RESIZE[1]}'}")
    
    # ==================== 数据加载器创建 ====================
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    print(f"批次大小: {BATCH_SIZE}, 梯度累积步数: {GRADIENT_ACCUMULATION_STEPS} (等效batch_size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    
    # ==================== 数据维度检查 ====================
    print("检查数据维度...")
    try:
        sample_batch = next(iter(train_loader))
        print('样本批次形状:', [tuple(x.shape) for x in sample_batch])
        print('数据格式: [batch_size, channels, height, width]')
    except Exception as e:
        print('获取样本批次失败:', e)
        return
    
    # ==================== 计算梯度判别模块启用epoch ====================
    # 如果未指定梯度判别模块启用epoch，则使用默认值（总训练轮数的40%）
    if ENABLE_GRADIENT_DISCRIMINATOR_EPOCH is None:
        enable_gd_epoch = int(EPOCHS * 0.4)
    else:
        enable_gd_epoch = ENABLE_GRADIENT_DISCRIMINATOR_EPOCH
    
    # ==================== 模型创建 ====================
    print("创建Y-FFC网络模型...")
    # 初始化时根据配置决定是否启用梯度判别模块
    initial_enable_gd = (enable_gd_epoch == 0)
    model = YFFC(enable_gradient_discriminator=initial_enable_gd, 
                gradient_constraint_weight=GRADIENT_CONSTRAINT_WEIGHT)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    if enable_gd_epoch > 0:
        print(f"梯度判别模块将在第 {enable_gd_epoch} 个epoch启用（默认值：总训练轮数的40%）")
    else:
        print("梯度判别模块已启用")
    print(f"梯度约束权重: {GRADIENT_CONSTRAINT_WEIGHT} (弱约束)")
    
    # ==================== 准备配置字典 ====================
    optimizer_config = {
        'type': OPTIMIZER_TYPE,
        'betas': ADAM_BETAS,
        'eps': ADAM_EPS,
        'weight_decay': ADAM_WEIGHT_DECAY,
        'momentum': SGD_MOMENTUM
    }
    
    scheduler_config = {
        'type': SCHEDULER_TYPE,
        'step_size': STEP_SIZE,
        'gamma': GAMMA,
        'T_max': T_MAX,
        'eta_min': ETA_MIN,
        'patience': PATIENCE,
        'factor': PLATEAU_FACTOR
    }
    
    loss_config = {
        'ssim_window_size': SSIM_WINDOW_SIZE,
        'ssim_sigma': SSIM_SIGMA,
        'weight_diffuse': LOSS_WEIGHT_DIFFUSE,
        'weight_specular': LOSS_WEIGHT_SPECULAR,
        'weight_gradient': LOSS_WEIGHT_GRADIENT
    }
    
    training_config = {
        'use_grad_clip': USE_GRAD_CLIP,
        'grad_clip_value': GRAD_CLIP_VALUE,
        'use_early_stopping': USE_EARLY_STOPPING,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'model_save_path': MODEL_SAVE_PATH,
        'checkpoint_dir': CHECKPOINT_DIR,
        'save_every_n_epochs': SAVE_EVERY_N_EPOCHS,
        'enable_gradient_discriminator_epoch': enable_gd_epoch,
        'gradient_constraint_weight': GRADIENT_CONSTRAINT_WEIGHT
    }
    
    # ==================== 模型训练 ====================
    print("\n" + "=" * 40)
    print("开始模型训练...")
    print("=" * 40)
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE,
        device=device,
        use_amp=USE_AMP,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        loss_config=loss_config,
        training_config=training_config
    )
    
    # ==================== 模型测试 ====================
    print("\n" + "=" * 40)
    print("开始模型测试...")
    print("=" * 40)
    # 加载训练过程中保存的最佳模型权重
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_model(model, test_loader, device=device)
    
    # ==================== 结果可视化 ====================
    print("\n" + "=" * 40)
    print("开始结果可视化...")
    print("=" * 40)
    visualize_results(model, test_loader, device=device, num_samples=VISUALIZATION_SAMPLES, output_folder=OUTPUT_FOLDER)
    
    # ==================== 训练历史可视化 ====================
    print("保存最终训练历史曲线...")
    
    # 创建2x2子图：损失和SSIM
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    epochs_range = range(1, len(train_losses) + 1)
    
    # 左上：训练和验证损失
    ax1.plot(epochs_range, train_losses, 'b-o', label='训练损失', linewidth=2, markersize=3)
    ax1.plot(epochs_range, val_losses, 'r-s', label='验证损失', linewidth=2, markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失值 (Loss)', fontsize=12)
    ax1.set_title('训练和验证损失', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右上：训练和验证SSIM
    ax2.plot(epochs_range, train_accuracies, 'b-o', label='训练SSIM', linewidth=2, markersize=3)
    ax2.plot(epochs_range, val_accuracies, 'r-s', label='验证SSIM', linewidth=2, markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('SSIM值', fontsize=12)
    ax2.set_title('训练和验证SSIM', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 左下：损失对比
    ax3.plot(epochs_range, train_losses, 'b-', label='训练损失', linewidth=2, alpha=0.7)
    ax3.plot(epochs_range, val_losses, 'r-', label='验证损失', linewidth=2, alpha=0.7)
    ax3.fill_between(epochs_range, train_losses, val_losses, alpha=0.2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('损失值 (Loss)', fontsize=12)
    ax3.set_title('损失对比（填充区域）', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 右下：SSIM对比
    ax4.plot(epochs_range, train_accuracies, 'b-', label='训练SSIM', linewidth=2, alpha=0.7)
    ax4.plot(epochs_range, val_accuracies, 'r-', label='验证SSIM', linewidth=2, alpha=0.7)
    ax4.fill_between(epochs_range, train_accuracies, val_accuracies, alpha=0.2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('SSIM值', fontsize=12)
    ax4.set_title('SSIM对比（填充区域）', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    fig.suptitle('Y-FFC Net 训练历史总览', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存到输出文件夹
    history_path = os.path.join(OUTPUT_FOLDER, 'training_history.png')
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练历史曲线已保存: {history_path}")
    
    # ==================== 生成训练总结报告 ====================
    summary_path = os.path.join(OUTPUT_FOLDER, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Y-FFC Net 训练总结报告\n")
        f.write("=" * 60 + "\n\n")
        f.write("训练配置:\n")
        f.write(f"  - 训练轮数: {EPOCHS}\n")
        f.write(f"  - 学习率: {LEARNING_RATE}\n")
        f.write(f"  - 批次大小: {BATCH_SIZE}\n")
        f.write(f"  - 梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"  - 图像尺寸: {IMAGE_RESIZE}\n")
        f.write(f"  - 优化器: {OPTIMIZER_TYPE}\n")
        f.write(f"  - 学习率调度器: {SCHEDULER_TYPE}\n")
        f.write(f"  - 混合精度训练: {'启用' if USE_AMP else '禁用'}\n\n")
        f.write("训练结果:\n")
        f.write(f"  - 最终训练损失: {train_losses[-1]:.6f}\n")
        f.write(f"  - 最终验证损失: {val_losses[-1]:.6f}\n")
        f.write(f"  - 最佳验证损失: {min(val_losses):.6f}\n")
        f.write(f"  - 最终训练SSIM: {train_accuracies[-1]:.4f}\n")
        f.write(f"  - 最终验证SSIM: {val_accuracies[-1]:.4f}\n")
        f.write(f"  - 最佳验证SSIM: {max(val_accuracies):.4f}\n\n")
        f.write("输出文件:\n")
        f.write(f"  - 最佳模型: {MODEL_SAVE_PATH}\n")
        f.write(f"  - 训练历史: {history_path}\n")
        f.write(f"  - 可视化结果: {os.path.join(OUTPUT_FOLDER, 'visualizations')}\n")
        if SAVE_EVERY_N_EPOCHS > 0:
            f.write(f"  - 检查点: {CHECKPOINT_DIR}\n")
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"\n训练总结报告已保存: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✅ 所有流程完成！")
    print("=" * 60)
    print(f"\n📁 所有结果已保存到: {os.path.abspath(OUTPUT_FOLDER)}")
    print("\n文件结构:")
    print(f"  {OUTPUT_FOLDER}/")
    print(f"  ├── models/                 (模型文件)")
    print(f"  │   └── best_model.pth      (最佳模型)")
    print(f"  ├── logs/                   (日志文件)")
    print(f"  ├── visualizations/         (可视化结果)")
    print(f"  │   ├── results_sample_1.png")
    print(f"  │   ├── results_sample_2.png")
    print(f"  │   └── results_sample_3.png")
    print(f"  ├── training_history.png    (训练曲线)")
    print(f"  └── training_summary.txt    (训练总结)")
    if SAVE_EVERY_N_EPOCHS > 0:
        print(f"  └── checkpoints/            (训练检查点)")
    print("=" * 60)

if __name__ == '__main__':  # 脚本入口
    # ========== Matplotlib 字体设置 ==========
    # 解决中文显示为乱码以及坐标轴负号显示问题
    # Windows 常见可用中文字体：Microsoft YaHei / SimHei
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
    rcParams['axes.unicode_minus'] = False
    main()  # 运行主流程