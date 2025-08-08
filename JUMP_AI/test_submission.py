import sys
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations

warnings.filterwarnings("ignore")

CFG = {
    "NBITS": 2048,
    "SEED": 42,
    "N_FOLDS": 5,
    "N_TRIALS": 5,
    "ENSEMBLE_WEIGHTS": [0.4, 0.3, 0.3],  # LGB, XGB, Cat 순서
}

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")

# npy 파일 경로 설정
X_train_path = os.path.join(data_dir, "X_train.npy")
X_test_path = os.path.join(data_dir, "X_test.npy")
y_train_path = os.path.join(data_dir, "y_train.npy")
sample_submission_path = os.path.join(data_dir, "sample_submission.csv")
trained_models_path = os.path.join(data_dir, "trained_models.pkl")


def check_paths():
    paths_to_check = [
        (X_train_path, "X_train.npy"),
        (X_test_path, "X_test.npy"),
        (y_train_path, "y_train.npy"),
        (sample_submission_path, "sample_submission.csv"),
        (trained_models_path, "trained_models.pkl")
    ]

    missing_files = []
    for path, filename in paths_to_check:
        if not os.path.exists(path):
            missing_files.append(filename)

    if missing_files:
        print(f"Warning: Missing files: {missing_files}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for data in: {data_dir}")
        return False
    return True


def load_npy_data():
    """npy 파일들을 로드합니다."""

    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)

    return X_train, X_test, y_train


def pIC50_to_IC50(pIC50):
    return 10 ** (9 - pIC50)


def IC50_to_pIC50(ic50_nM):
    ic50_nM = np.clip(ic50_nM, 1e-10, None)
    return 9 - np.log10(ic50_nM)


def calculate_leaderboard_score(y_true, y_pred):
    # Component A: Normalized RMSE (IC50 nM)
    ic50_true = pIC50_to_IC50(y_true)
    ic50_pred = pIC50_to_IC50(y_pred)

    rmse_ic50 = np.sqrt(mean_squared_error(ic50_true, ic50_pred))
    normalized_rmse = rmse_ic50 / (np.max(ic50_true) - np.min(ic50_true))
    component_a = min(normalized_rmse, 1.0)
    component_b = r2_score(y_true, y_pred)
    score = 0.4 * (1 - component_a) + 0.6 * component_b

    return {
        'component_a': component_a,
        'component_b': component_b,
        'score': score,
        'rmse_ic50': rmse_ic50,
        'normalized_rmse': normalized_rmse,
    }


def simple_ensemble(predictions, weights=None):
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)

    ensemble_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        ensemble_pred += weight * pred

    return ensemble_pred


def calculate_model_weights(models, model_names, X_train, y_train, n_folds=5, seed=42):
    """개별 모델의 성능에 따라 가중치를 계산합니다."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    model_scores = {}

    for model_name in model_names:
        model_scores[model_name] = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model = models[model_name][fold]
            val_pred = model.predict(X_val_fold)

            # 리더보드 스코어 계산
            results = calculate_leaderboard_score(y_val_fold, val_pred)
            model_scores[model_name].append(results['score'])

    # 각 모델의 평균 스코어 계산
    avg_scores = {}
    for model_name in model_names:
        avg_scores[model_name] = np.mean(model_scores[model_name])

    # 스코어를 가중치로 변환 (스코어가 높을수록 가중치가 큼)
    total_score = sum(avg_scores.values())
    weights = [avg_scores[model_name] /
               total_score for model_name in model_names]

    print(f"Model weights based on performance:")
    for model_name, weight in zip(model_names, weights):
        print(
            f"  {model_name}: {weight:.4f} (avg score: {avg_scores[model_name]:.4f})")

    return weights


def evaluate_ensemble_combination(models, model_names, ensemble_method, X_train, y_train, n_folds=5, seed=42):
    """특정 모델 조합과 앙상블 방법으로 CV 스코어를 계산합니다."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_val_preds = []
    all_val_trues = []

    # 가중치 계산
    if ensemble_method == "dynamic_simple":
        weights = calculate_model_weights(
            models, model_names, X_train, y_train, n_folds, seed)
    elif ensemble_method == "equal_weight":
        weights = [1.0 / len(model_names)] * len(model_names)
        print(f"Equal weights: {weights}")
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        fold_predictions = []

        for model_name in model_names:
            model = models[model_name][fold]
            val_pred = model.predict(X_val_fold)
            fold_predictions.append(val_pred)

        # 가중치를 사용한 simple ensemble
        ensemble_pred = simple_ensemble(fold_predictions, weights)

        all_val_preds.extend(ensemble_pred)
        all_val_trues.extend(y_val_fold)

    all_val_preds = np.array(all_val_preds)
    all_val_trues = np.array(all_val_trues)

    # 리더보드 스코어 계산
    results = calculate_leaderboard_score(all_val_trues, all_val_preds)

    return results['score'], results


def test_all_combinations():
    """모든 모델 조합과 앙상블 방법을 테스트합니다."""
    # 데이터 로드
    X_train, X_test, y_train = load_npy_data()

    # 훈련된 모델 로드
    with open(trained_models_path, 'rb') as f:
        model_data = pickle.load(f)
        models = model_data['models']

    print(f"Loaded {len(models)} model types: {list(models.keys())}")

    # 가능한 모델 조합 생성 (단일 모델 제외)
    model_types = list(models.keys())  # ['lgb', 'xgb', 'cat']
    all_combinations = []

    # 2개, 3개 모델 조합만 (단일 모델 제외)
    for r in range(2, len(model_types) + 1):
        for combo in combinations(model_types, r):
            all_combinations.append(list(combo))

    print(f"Testing {len(all_combinations)} model combinations:")
    for combo in all_combinations:
        print(f"  - {combo}")

    # 앙상블 방법들 (동적 가중치와 균등 가중치 비교)
    ensemble_methods = ["dynamic_simple", "equal_weight"]

    # 결과 저장
    best_score = -1
    best_config = None
    best_results = None
    all_results = []

    # 모든 조합 테스트
    for combo in all_combinations:
        for method in ensemble_methods:
            print(f"\nTesting combination: {combo} with {method} ensemble")

            try:
                score, results = evaluate_ensemble_combination(
                    models, combo, method, X_train, y_train, CFG["N_FOLDS"], CFG["SEED"]
                )

                result_info = {
                    'combination': combo,
                    'method': method,
                    'score': score,
                    'results': results
                }
                all_results.append(result_info)

                print(f"  Score: {score:.4f}")
                print(f"  Component A: {results['component_a']:.4f}")
                print(f"  Component B: {results['component_b']:.4f}")

                if score > best_score:
                    best_score = score
                    best_config = (combo, method)
                    best_results = results
                    print(f"  *** NEW BEST! ***")

            except Exception as e:
                print(f"  Error: {e}")
                continue

    # 결과 정렬 및 출력
    all_results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*60}")
    print("TOP 10 RESULTS:")
    print(f"{'='*60}")

    for i, result in enumerate(all_results[:10]):
        combo_str = '+'.join(result['combination'])
        print(
            f"{i+1:2d}. {combo_str:15s} | {result['method']:8s} | {result['score']:.4f}")

    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION:")
    print(f"{'='*60}")
    print(f"Models: {best_config[0]}")
    print(f"Method: {best_config[1]}")
    print(f"Score: {best_score:.4f}")
    print(f"Component A: {best_results['component_a']:.4f}")
    print(f"Component B: {best_results['component_b']:.4f}")

    return best_config, best_results, models


def generate_final_predictions(best_config, models):
    """최고 성능 조합으로 최종 예측을 생성합니다."""
    print(f"\nGenerating final predictions with best configuration...")

    X_train, X_test, y_train = load_npy_data()
    best_models, best_method = best_config

    # 가중치 계산
    if best_method == "dynamic_simple":
        weights = calculate_model_weights(
            models, best_models, X_train, y_train, CFG["N_FOLDS"], CFG["SEED"])
    elif best_method == "equal_weight":
        weights = [1.0 / len(best_models)] * len(best_models)
        print(f"Using equal weights: {weights}")
    else:
        raise ValueError(f"Unknown ensemble method: {best_method}")

    # 테스트 데이터에 대한 예측
    test_predictions = np.zeros(len(X_test))

    for fold in range(CFG['N_FOLDS']):
        fold_predictions = []

        for model_name in best_models:
            model = models[model_name][fold]
            fold_pred = model.predict(X_test)
            fold_predictions.append(fold_pred)

        # 가중치를 사용한 simple ensemble
        ensemble_pred = simple_ensemble(fold_predictions, weights)
        test_predictions += ensemble_pred

    test_predictions /= CFG['N_FOLDS']

    print(f"Test predictions shape: {test_predictions.shape}")

    return test_predictions


def save_submission(test_predictions):
    """예측 결과를 sample_submission 형식으로 저장합니다."""

    # sample_submission.csv 로드
    submission = pd.read_csv(sample_submission_path)

    # pIC50을 IC50 nM으로 변환
    ic50_predictions = pIC50_to_IC50(test_predictions)

    # submission에 예측값 저장
    submission["ASK1_IC50_nM"] = ic50_predictions

    # 결과 저장
    submission.to_csv("JUMP_AI/data/final_submission.csv", index=False)


if __name__ == "__main__":
    # 모든 조합 테스트
    best_config, best_results, models = test_all_combinations()

    # 최종 예측 생성
    test_predictions = generate_final_predictions(best_config, models)

    # 결과 저장
    save_submission(test_predictions)

    print(f"Best configuration: {best_config}")
    print(f"Best score: {best_results['score']:.4f}")
