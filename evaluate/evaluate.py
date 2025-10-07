def calculate_pixel_diff(self, orig_img, adv_img):
    """计算像素级差异指标"""
    
    orig = np.array(orig_img).astype(float)
    adv = np.array(adv_img).astype(float)
    diff = np.abs(orig - adv)
    
    metrics = {
        "MAE": np.mean(diff),
        "MSE": mean_squared_error(orig, adv),
        "PSNR": peak_signal_noise_ratio(orig, adv, data_range=255),
        "Max Pixel Diff": np.max(diff),
        "Pixels Changed (>5)": np.mean(diff > 5) * 100,
        "L2 Norm": np.linalg.norm(orig - adv) / np.linalg.norm(orig)
    }
    return metrics

def calculate_ssim(self, orig_img, adv_img):
    """计算结构相似性指数"""
    orig = np.array(orig_img)
    adv = np.array(adv_img)
    return structural_similarity(orig, adv, channel_axis=2)

def calculate_lpips(self, orig_img, adv_img):
    """计算感知差异 (LPIPS)"""
    if not self.LPIPS_AVAILABLE:
        return None
    
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        orig_tensor = transform(orig_img).unsqueeze(0)
        adv_tensor = transform(adv_img).unsqueeze(0)
        
        if torch.cuda.is_available():
            orig_tensor = orig_tensor.cuda()
            adv_tensor = adv_tensor.cuda()
        
        with torch.no_grad():
            return self.LPIPS_MODEL(orig_tensor, adv_tensor).item()
    except Exception as e:
        logger.error(f"计算LPIPS失败: {str(e)}")
        return None

def calculate_asr(self, orig_predictions, adv_predictions, target_labels=None):
    """
    计算攻击成功率 (Attack Success Rate, ASR)
    
    :param orig_predictions: 原始图像的预测结果
    :param adv_predictions: 对抗样本的预测结果
    :param target_labels: 目标标签（针对目标攻击，可选）
    :return: 攻击成功率
    """
    orig_predictions = np.array(orig_predictions)
    adv_predictions = np.array(adv_predictions)
    
    if target_labels is not None:
        # 目标攻击：攻击成功是指模型预测为目标标签
        target_labels = np.array(target_labels)
        success_count = np.sum(adv_predictions == target_labels)
    else:
        # 非目标攻击：攻击成功是指模型预测结果与原始预测不同
        success_count = np.sum(orig_predictions != adv_predictions)
    
    return success_count / len(orig_predictions)

def calculate_transfer_success_rate(self, source_model_predictions, target_model_predictions, 
                                    original_labels):
    """
    计算迁移攻击成功率 (Transfer Success Rate)
    
    :param source_model_predictions: 源模型对对抗样本的预测结果
    :param target_model_predictions: 目标模型对对抗样本的预测结果
    :param original_labels: 原始标签
    :return: 迁移攻击成功率
    """
    source_model_predictions = np.array(source_model_predictions)
    target_model_predictions = np.array(target_model_predictions)
    original_labels = np.array(original_labels)
    
    # 在源模型上攻击成功的样本
    source_attack_success = source_model_predictions != original_labels
    # 在这些样本中，目标模型上也攻击成功的比例
    transfer_success = np.sum(
        (source_model_predictions != original_labels) & 
        (target_model_predictions != original_labels)
    )
    
    if np.sum(source_attack_success) > 0:
        return transfer_success / np.sum(source_attack_success)
    else:
        return 0.0

def calculate_l0_norm(self, orig_img, adv_img):
    """
    计算L0范数（修改的像素数量）
    
    :param orig_img: 原始图像
    :param adv_img: 对抗样本图像
    :return: L0范数
    """
    orig = np.array(orig_img).astype(float)
    adv = np.array(adv_img).astype(float)
    diff = np.abs(orig - adv)
    # 计算非零像素的数量（差异大于某个小阈值）
    return np.sum(diff > 1e-6)

def calculate_l1_norm(self, orig_img, adv_img):
    """
    计算L1范数（扰动能量）
    
    :param orig_img: 原始图像
    :param adv_img: 对抗样本图像
    :return: L1范数
    """
    orig = np.array(orig_img).astype(float)
    adv = np.array(adv_img).astype(float)
    diff = np.abs(orig - adv)
    return np.sum(diff)

def calculate_l2_norm(self, orig_img, adv_img):
    """
    计算L2范数（扰动能量）
    
    :param orig_img: 原始图像
    :param adv_img: 对抗样本图像
    :return: L2范数
    """
    orig = np.array(orig_img).astype(float)
    adv = np.array(adv_img).astype(float)
    diff = orig - adv
    return np.linalg.norm(diff)

def calculate_linf_norm(self, orig_img, adv_img):
    """
    计算L∞范数（最大单像素扰动）
    
    :param orig_img: 原始图像
    :param adv_img: 对抗样本图像
    :return: L∞范数
    """
    orig = np.array(orig_img).astype(float)
    adv = np.array(adv_img).astype(float)
    diff = np.abs(orig - adv)
    return np.max(diff)

def calculate_psnr(self, orig_img, adv_img):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)
    
    :param orig_img: 原始图像
    :param adv_img: 对抗样本图像
    :return: PSNR值
    """
    orig = np.array(orig_img).astype(float)
    adv = np.array(adv_img).astype(float)
    mse = np.mean((orig - adv) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ciede2000(self, orig_img, adv_img):
    """
    计算人眼色差 ΔE (CIEDE2000)
    
    :param orig_img: 原始图像
    :param adv_img: 对抗样本图像
    :return: 平均色差值
    """
    try:
        from colormath.color_objects import sRGBColor, LabColor
        from colormath.color_conversions import convert_color
        from colormath.color_diff import delta_e_cie2000
    except ImportError:
        logger.warning("colormath库未安装，无法计算CIEDE2000色差")
        return None
    
    orig = np.array(orig_img)
    adv = np.array(adv_img)
    
    delta_e_values = []
    for i in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            orig_pixel = orig[i, j] / 255.0
            adv_pixel = adv[i, j] / 255.0
            
            # 转换为Lab颜色空间
            orig_rgb = sRGBColor(orig_pixel[0], orig_pixel[1], orig_pixel[2])
            adv_rgb = sRGBColor(adv_pixel[0], adv_pixel[1], adv_pixel[2])
            
            orig_lab = convert_color(orig_rgb, LabColor)
            adv_lab = convert_color(adv_rgb, LabColor)
            
            # 计算色差
            delta_e = delta_e_cie2000(orig_lab, adv_lab)
            delta_e_values.append(delta_e)
    
    return np.mean(delta_e_values) if delta_e_values else 0.0

# def calculate_fid(self, orig_images, adv_images):
#     """
#     计算图像分布差异 (Fréchet Inception Distance, FID)
    
#     :param orig_images: 原始图像列表
#     :param adv_images: 对抗样本图像列表
#     :return: FID值
#     """
#     # 注意：此方法需要预先加载Inception模型，此处仅为框架
#     logger.warning("FID计算需要额外的Inception模型，此处仅提供框架")
#     return None

def calculate_robust_accuracy(self, orig_predictions, adv_predictions, original_labels, epsilon):
    """
    计算鲁棒精度 (Robust Accuracy@ε)
    
    :param orig_predictions: 原始图像的预测结果
    :param adv_predictions: 对抗样本的预测结果
    :param original_labels: 真实标签
    :param epsilon: 扰动半径
    :return: 鲁棒精度
    """
    orig_predictions = np.array(orig_predictions)
    adv_predictions = np.array(adv_predictions)
    original_labels = np.array(original_labels)
    
    # 原始预测正确的样本
    correct_orig = orig_predictions == original_labels
    # 对抗样本预测错误的样本
    incorrect_adv = adv_predictions != original_labels
    # 在扰动半径ε下的鲁棒精度：原始正确且对抗样本也正确的比例
    robust_correct = correct_orig & ~incorrect_adv
    
    if np.sum(correct_orig) > 0:
        return np.sum(robust_correct) / np.sum(correct_orig)
    else:
        return 0.0

def calculate_query_count(self, query_counts):
    """
    计算平均查询次数
    
    :param query_counts: 每个样本的查询次数列表
    :return: 平均查询次数
    """
    query_counts = np.array(query_counts)
    return np.mean(query_counts)

def calculate_accuracy_drop(self, orig_predictions, adv_predictions, original_labels):
    """
    计算模型准确率下降
    
    :param orig_predictions: 原始图像的预测结果
    :param adv_predictions: 对抗样本的预测结果
    :param original_labels: 真实标签
    :return: 准确率下降幅度
    """
    orig_predictions = np.array(orig_predictions)
    adv_predictions = np.array(adv_predictions)
    original_labels = np.array(original_labels)
    
    orig_accuracy = np.sum(orig_predictions == original_labels) / len(original_labels)
    adv_accuracy = np.sum(adv_predictions == original_labels) / len(original_labels)
    
    return orig_accuracy - adv_accuracy

def calculate_backdoor_detection_rate(self, detected_backdoor_count, total_backdoor_samples):
    """
    计算后门检测成功率 (Detection Success Rate, DSR)
    
    :param detected_backdoor_count: 检测到的后门样本数
    :param total_backdoor_samples: 总的后门样本数
    :return: 后门检测成功率
    """
    if total_backdoor_samples > 0:
        return detected_backdoor_count / total_backdoor_samples
    else:
        return 0.0

def calculate_performance_degradation(self, orig_metric, adv_metric):
    """
    计算性能下降幅度（攻击伤害/防御增益）
    
    :param orig_metric: 原始模型的性能指标值
    :param adv_metric: 攻击后/防御后的性能指标值
    :return: 性能下降幅度
    """
    return orig_metric - adv_metric
