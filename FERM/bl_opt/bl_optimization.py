import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from numpy.linalg import solve


def mean_var_opt(mu, cov_matrix, risk_aversion, long_only=True, weight_bounds=(0.001, 0.25)):
    '''
    cvxopt 양식:
        min (1/2) wᵀ (δΣ) w + (-μᵀ) w
        s.t. 1ᵀ w = 1
            (-I)w ≤ 0 (no shortselling)
    long_only: 롱온리 제약 여부
    return: (BL 사후 분포 만족하는) 가중치
    '''
    n = len(mu)  # 자산 개수
    # 공분산 singular 방지
    eps = 1e-8

    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    if isinstance(mu, (pd.Series, pd.DataFrame)):
        mu = np.array(mu).ravel()

    P = matrix(risk_aversion * (cov_matrix + eps * np.eye(n)))  # n*n
    q = matrix(-mu, (n, 1))  # n*1

    # 부등식 제약. G = -I, h = [0,..,0]ᵀ
    G_list, h_list = [], []
    if long_only:
        G_list.append(-np.eye(n))
        h_list.append(np.zeros(n))
    if weight_bounds is not None:
        G_list.append(np.eye(n))
        h_list.append(np.full(n, weight_bounds[1]))
        # w_i ≥ w_min  →  -w_i ≤ -w_min
        G_list.append(-np.eye(n))
        h_list.append(-np.full(n, weight_bounds[0]))

    G, h = (matrix(np.vstack(G_list), tc='d'), matrix(
        np.concatenate(h_list), tc='d')) if len(G_list) > 0 else (None, None)

    # 등식 제약
    A = matrix(np.ones(n), (1, n))  # A = [1,1,...,1]
    b = matrix(1.0)

    # Optimization
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    w = np.array(sol['x']).ravel()
    return w


def max_sharpe_opt(mu, cov_matrix, risk_free_rate=0.0, long_only=True, w_max=None):
    """
    Maximize Sharpe Ratio (Tangency Portfolio)
    using CVXOPT quadratic programming.

    max (mu - rf)' w
    s.t. w' Σ w <= 1
         sum(w) = 1 (optional, can be omitted)
         w >= 0 (if long_only)

    Returns: normalized tangency portfolio weights
    """
    n = len(mu)
    eps = 1e-8

    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    if isinstance(mu, (pd.Series, pd.DataFrame)):
        mu = np.array(mu).ravel()

    # Excess returns
    excess_mu = mu - risk_free_rate

    # QP form: minimize (1/2) w' P w + q' w
    # But we maximize Sharpe ⇒ minimize -excess_mu' w
    P = matrix(cov_matrix + eps * np.eye(n))
    q = matrix(-excess_mu, (n, 1))

    # Linear constraints: sum(w)=1, w>=0, w<=w_max
    G_list, h_list = [], []
    if long_only:
        G_list.append(-np.eye(n))
        h_list.append(np.zeros(n))
    if w_max is not None:
        G_list.append(np.eye(n))
        h_list.append(np.full(n, w_max))

    G = matrix(np.vstack(G_list)) if G_list else None
    h = matrix(np.concatenate(h_list)) if h_list else None

    A = matrix(np.ones((1, n)))
    b = matrix(1.0)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    w_raw = np.array(sol['x']).ravel()

    # Normalize weights to sum to 1 (portfolio constraint)
    w = w_raw / w_raw.sum()

    return w
