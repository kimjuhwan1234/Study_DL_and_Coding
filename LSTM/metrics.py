import torch
import numpy as np


def ensure_tensor_array(x):
    """입력이 Tensor / List / 단일 값 / 이중 리스트이면 numpy 배열로 변환"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()  # Tensor → numpy 변환
    elif isinstance(x, list):
        x = np.array(x)  # 리스트 → numpy 변환

    if x.ndim == 2 and x.shape[1] == 1:  # (N,1) 형태 → (N,) 변환
        x = x.squeeze()

    return np.array(x) if isinstance(x, np.ndarray) else np.array([x])  # 단일 값이면 배열 변환


def ensure_binary_labels(x):
    """0 또는 1의 정수형인지 확인하고 변환"""
    x = ensure_tensor_array(x)  # numpy 변환
    return (x >= 0.5).astype(int) if not np.issubdtype(x.dtype, np.integer) else x


# 🔹 회귀 평가 함수 (MSE, RMSE, R2 등)
def MAE(output, gt):
    """Mean Absolute Error (MAE) 계산"""
    return np.mean(np.abs(ensure_tensor_array(output) - ensure_tensor_array(gt)))


def MSE(output, gt):
    """Mean Squared Error (MSE) 계산"""
    return np.mean((ensure_tensor_array(output) - ensure_tensor_array(gt)) ** 2)


def RMSE(output, gt):
    """Root Mean Squared Error (RMSE) 계산"""
    return np.sqrt(MSE(output, gt))


def R2(output, gt):
    """R-squared (R²) 계산"""
    output, gt = ensure_tensor_array(output), ensure_tensor_array(gt)
    mean_gt = np.mean(gt)
    ss_total = np.sum((gt - mean_gt) ** 2)
    ss_residual = np.sum((output - gt) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total != 0 else 0


def AdjustedR2(output, gt, n, p):
    """Adjusted R-squared 계산"""
    r2 = R2(output, gt)
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else r2


# 🔹 분류 평가 함수 (Precision, Recall, F1 Score)
def Precision(output, gt):
    """ 다중 클래스 Precision (벡터 연산 적용) """
    output, gt = ensure_tensor_array(output), ensure_tensor_array(gt)

    # 예측값이 확률 벡터이면 argmax 적용
    if output.ndim > 1 and output.shape[1] > 1:
        output = np.argmax(output, axis=1)

    num_classes = np.unique(gt).size  # ✅ 더 빠르게 클래스 개수 계산

    TP = np.array([(output == k) & (gt == k) for k in range(num_classes)]).sum(axis=1)
    FP = np.array([(output == k) & (gt != k) for k in range(num_classes)]).sum(axis=1)

    precision = np.divide(TP, (TP + FP), where=(TP + FP) > 0, out=np.zeros_like(TP, dtype=float))

    return np.mean(precision)  # ✅ macro-average 반환 (벡터 연산)


def Recall(output, gt):
    """ 다중 클래스 Recall (벡터 연산 적용) """
    output, gt = ensure_tensor_array(output), ensure_tensor_array(gt)

    if output.ndim > 1 and output.shape[1] > 1:
        output = np.argmax(output, axis=1)

    num_classes = np.unique(gt).size  # ✅ 더 빠르게 클래스 개수 계산

    TP = np.array([(output == k) & (gt == k) for k in range(num_classes)]).sum(axis=1)
    FN = np.array([(output != k) & (gt == k) for k in range(num_classes)]).sum(axis=1)

    recall = np.divide(TP, (TP + FN), where=(TP + FN) > 0, out=np.zeros_like(TP, dtype=float))

    return np.mean(recall)  # ✅ macro-average 반환 (벡터 연산)


def F1Score(output, gt):
    """ 다중 클래스 F1 Score (벡터 연산 적용) """
    precision = Precision(output, gt)
    recall = Recall(output, gt)

    return np.divide(2 * precision * recall, (precision + recall), where=(precision + recall) > 0,
                     out=np.zeros_like(precision, dtype=float))
