import warnings
import numpy as np
import pandas as pd
from numpy.linalg import solve
from sklearn.linear_model import ElasticNet


def Pi(market_caps, risk_aversion, cov_matrix):  # risk_free_rate은 0으로 고정
    """
    market weight기반으로 implied equilibrium excess return(pi)를 구함
    Π = δΣwmkt

    Parameters:
    - market_caps: 각 자산의 시가총액. {ticker: cap} dict or pd.Series
    - risk_aversion: 위험 회피 계수. positive float
    - cov_matrix: covariance matrix of asset returns. pd.DataFrame

    Returns:
    - implied equilibrium excess return. pd.Series
    """
    if not isinstance(cov_matrix, pd.DataFrame):
        warnings.warn(
            "If cov_matrix is not a dataframe, market cap index must be aligned to cov_matrix",
            RuntimeWarning,
        )
    market_caps_series = pd.Series(market_caps)
    mkt_w = market_caps_series / market_caps_series.sum()

    return risk_aversion * cov_matrix.dot(mkt_w)


def posterior_dist(cov_matrix, pi, P=None, Q=None, Omega=None, tau=0.025):
    """
    return: μ_BL, Σ_BL
    μ_BL = M⁻¹ * b,  where
        M = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P
        b = (τΣ)⁻¹ π + Pᵀ Ω⁻¹ q
    Σ_BL = Σ + (M)⁻¹
    """
    if P is None or Q is None:
        # 전망이 없으면 prior distribution만
        return pi.copy(), cov_matrix.copy()

    # (τΣ)⁻¹ 계산
    I = np.eye(cov_matrix.shape[0])
    inv_ts = solve(tau * cov_matrix, I)

    # Ω⁻¹ 계산
    I_omega = np.eye(Omega.shape[0])
    inv_omega = solve(Omega, I_omega)

    P_inv_om = P.T @ inv_omega  # Pᵀ Ω⁻¹

    M = inv_ts + P_inv_om @ P
    b = inv_ts @ pi + P_inv_om @ Q
    inv_M = solve(M, I)

    mu_bl = solve(M, b)
    sigma_bl = cov_matrix + inv_M
    return mu_bl, sigma_bl


# 안쓰는게 나음
def posterior_dist_elastic(
    cov_matrix, pi, P=None, Q=None, Omega=None, tau=0.025,
    lambda1=0.0, lambda2=0.0
):
    """
    Elastic Net 기반 Black–Litterman posterior (논문식 (4))
      min_μ (y-Bμ)^T V^{-1}(y-Bμ) + λ2||μ||_2^2 + λ1||μ||_1
    sklearn.ElasticNet 기반 구현
    반환값: μ_BL_enet, Σ_BL
    """
    if P is None or Q is None:
        return pi.copy(), cov_matrix.copy()

    eps = 1e-6
    inv_tauSigma = np.linalg.inv(
        tau * cov_matrix + eps * np.eye(cov_matrix.shape[0]))
    inv_Omega = np.linalg.inv(Omega + eps * np.eye(Omega.shape[0]))

    # V^{-1}
    W = np.block([
        [inv_tauSigma, np.zeros((inv_tauSigma.shape[0], inv_Omega.shape[1]))],
        [np.zeros((inv_Omega.shape[0], inv_tauSigma.shape[1])), inv_Omega]
    ])

    # 데이터 결합
    y = np.concatenate([pi, Q])
    B = np.vstack([np.eye(cov_matrix.shape[0]), P])

    # 3️⃣ Whitening (Cholesky)
    C = np.linalg.cholesky(W + 1e-8*np.eye(W.shape[0]))
    y_t = C @ y
    B_t = C @ B

    # Elastic Net 회귀 (논문식과 매칭)
    # alpha = λ1 + λ2, l1_ratio = λ1 / (λ1 + λ2)
    if lambda1 == 0 and lambda2 == 0:
        # classical BL 해
        I = np.eye(cov_matrix.shape[0])
        H = inv_tauSigma + P.T @ inv_Omega @ P
        b = inv_tauSigma @ pi + P.T @ (inv_Omega @ Q)
        mu_bl = solve(H, b)
        sigma_bl = cov_matrix + np.linalg.inv(H)
        return mu_bl, sigma_bl

    alpha = lambda1 + lambda2
    l1_ratio = lambda1 / alpha if alpha != 0 else 0.0

    reg = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        max_iter=10000,
        tol=1e-7
    )
    reg.fit(B_t, y_t)
    mu_bl = reg.coef_

    # 논문식 (5) - 기존 계산된 역행렬 재사용
    sigma_bl = cov_matrix + np.linalg.inv(inv_tauSigma + P.T @ inv_Omega @ P)
    return mu_bl, sigma_bl
