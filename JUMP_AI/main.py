import sys
import os
import pickle
import optuna
import warnings
import numpy as np
import pandas as pd
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

CFG = {
    "NBITS": 2048,
    "SEED": 42,
    "N_FOLDS": 5,
    "N_TRIALS": 5,  # Optuna 시도 횟수
    "ENSEMBLE_WEIGHTS": [0.4, 0.3, 0.3],  # LGB, XGB, Cat 순서
}

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")

# npy 파일 경로 설정
X_train_path = os.path.join(data_dir, "X_train.npy")
X_test_path = os.path.join(data_dir, "X_test.npy")
y_train_path = os.path.join(data_dir, "y_train.npy")


def check_paths():
    paths_to_check = [
        (X_train_path, "X_train.npy"),
        (X_test_path, "X_test.npy"),
        (y_train_path, "y_train.npy")
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
    print("Loading npy files...")

    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")

    return X_train, X_test, y_train


def pIC50_to_IC50(pIC50):
    return 10 ** (9 - pIC50)


def IC50_to_pIC50(ic50_nM):
    ic50_nM = np.clip(ic50_nM, 1e-10, None)
    return 9 - np.log10(ic50_nM)


def objective_lgb(trial, X, y, cv_scores, n_folds=5, seed=42):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": 1000,
        "random_state": seed,
        "verbose": -1,
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_val_preds = []
    all_val_trues = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        pred = model.predict(X_val)
        all_val_preds.extend(pred)
        all_val_trues.extend(y_val)

    all_val_preds = np.array(all_val_preds)
    all_val_trues = np.array(all_val_trues)

    results = calculate_leaderboard_score(all_val_trues, all_val_preds)
    cv_scores["lgb"].append(results['score'])
    return results['score']


def objective_xgb(trial, X, y, cv_scores, n_folds=5, seed=42):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "n_estimators": 1000,
        "random_state": seed,
        "verbosity": 0,
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_val_preds = []
    all_val_trues = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=0,
        )

        pred = model.predict(X_val)
        all_val_preds.extend(pred)
        all_val_trues.extend(y_val)

    all_val_preds = np.array(all_val_preds)
    all_val_trues = np.array(all_val_trues)

    results = calculate_leaderboard_score(all_val_trues, all_val_preds)
    cv_scores["xgb"].append(results['score'])
    return results['score']


def objective_cat(trial, X, y, cv_scores, n_folds=5, seed=42):
    params = {
        "loss_function": "RMSE",
        "iterations": 1000,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "random_state": seed,
        "verbose": False,
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_val_preds = []
    all_val_trues = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=False,
        )

        pred = model.predict(X_val)
        all_val_preds.extend(pred)
        all_val_trues.extend(y_val)

    all_val_preds = np.array(all_val_preds)
    all_val_trues = np.array(all_val_trues)

    results = calculate_leaderboard_score(all_val_trues, all_val_preds)
    cv_scores["cat"].append(results['score'])
    return results['score']


def simple_ensemble(predictions, weights=None):
    if weights is None:
        weights = CFG["ENSEMBLE_WEIGHTS"]

    ensemble_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        ensemble_pred += weight * pred

    return ensemble_pred


def rank_ensemble(predictions, weights=None):
    if weights is None:
        weights = CFG["ENSEMBLE_WEIGHTS"]

    ranked_preds = []
    for pred in predictions:
        ranked_pred = np.argsort(np.argsort(pred))
        ranked_preds.append(ranked_pred)

    ensemble_rank = np.zeros_like(ranked_preds[0])
    for rank_pred, weight in zip(ranked_preds, weights):
        ensemble_rank += weight * rank_pred

    return ensemble_rank


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


if __name__ == "__main__":
    print("JUMP AI - LightGBM, XGBoost, CatBoost Ensemble with Optuna")
    print("=" * 60)

    print("Checking file paths...")
    if not check_paths():
        print("Error: Required data files not found!")
        sys.exit(1)

    print(f"Data directory: {data_dir}")
    print("All required files found!")

    print("Loading data...")
    X_train, X_test, y_train = load_npy_data()

    print(
        f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    cv_scores = {"lgb": [], "xgb": [], "cat": []}




    print("\nTuning LightGBM hyperparameters...")
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(
        lambda trial: objective_lgb(
            trial, X_train, y_train, cv_scores, CFG["N_FOLDS"], CFG["SEED"]),
        n_trials=CFG["N_TRIALS"]
    )

    best_params_lgb = study_lgb.best_params
    print(f"LightGBM best parameters: {best_params_lgb}")
    print(f"LightGBM best leaderboard score: {study_lgb.best_value:.4f}")




    print("\nTuning XGBoost hyperparameters...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(
        lambda trial: objective_xgb(
            trial, X_train, y_train, cv_scores, CFG["N_FOLDS"], CFG["SEED"]),
        n_trials=CFG["N_TRIALS"]
    )

    best_params_xgb = study_xgb.best_params
    print(f"XGBoost best parameters: {best_params_xgb}")
    print(f"XGBoost best leaderboard score: {study_xgb.best_value:.4f}")




    print("\nTuning CatBoost hyperparameters...")
    study_cat = optuna.create_study(direction="maximize")
    study_cat.optimize(
        lambda trial: objective_cat(
            trial, X_train, y_train, cv_scores, CFG["N_FOLDS"], CFG["SEED"]),
        n_trials=CFG["N_TRIALS"]
    )

    best_params_cat = study_cat.best_params
    print(f"CatBoost best parameters: {best_params_cat}")
    print(f"CatBoost best leaderboard score: {study_cat.best_value:.4f}")





    print("\nTraining final models with optimal parameters...")

    kf = KFold(n_splits=CFG["N_FOLDS"], shuffle=True, random_state=CFG["SEED"])

    models = {
        'lgb': [],
        'xgb': [],
        'cat': []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Training Fold {fold + 1}/{CFG['N_FOLDS']}...")

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(
            X_val_fold, label=y_val_fold, reference=train_data)

        lgb_model = lgb.train(
            best_params_lgb,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        models['lgb'].append(lgb_model)

        xgb_model = xgb.XGBRegressor(**best_params_xgb)
        xgb_model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            early_stopping_rounds=100,
            verbose=0,
        )
        models['xgb'].append(xgb_model)

        cat_model = cb.CatBoostRegressor(**best_params_cat)
        cat_model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            early_stopping_rounds=100,
            verbose=False,
        )
        models['cat'].append(cat_model)





    print("\nSaving trained models...")
    with open('JUMP_AI/data/trained_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'best_params_lgb': best_params_lgb,
            'best_params_xgb': best_params_xgb,
            'best_params_cat': best_params_cat,
            'ensemble_weights': CFG["ENSEMBLE_WEIGHTS"]
        }, f)
    print("Models saved to trained_models.pkl")





    print("\nModel performance comparison:")
    print(
        f"LightGBM CV score: {np.mean(cv_scores['lgb']):.4f} ± {np.std(cv_scores['lgb']):.4f}"
    )
    print(
        f"XGBoost CV score: {np.mean(cv_scores['xgb']):.4f} ± {np.std(cv_scores['xgb']):.4f}"
    )
    print(
        f"CatBoost CV score: {np.mean(cv_scores['cat']):.4f} ± {np.std(cv_scores['cat']):.4f}"
    )





    print("\n3. Leaderboard Score Evaluation:")
    print("Collecting validation predictions for leaderboard score calculation...")

    all_val_preds = []
    all_val_trues = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        val_preds = np.zeros(len(X_val_fold))

        for model_type in ['lgb', 'xgb', 'cat']:
            if model_type == 'lgb':
                val_preds += models[model_type][fold].predict(X_val_fold)
            elif model_type == 'xgb':
                val_preds += models[model_type][fold].predict(X_val_fold)
            else:
                val_preds += models[model_type][fold].predict(X_val_fold)

        val_preds /= 3
        all_val_preds.extend(val_preds)
        all_val_trues.extend(y_val_fold)

    all_val_preds = np.array(all_val_preds)
    all_val_trues = np.array(all_val_trues)

    print(f"Total validation samples: {len(all_val_trues)}")
    print(
        f"Prediction range: {np.min(all_val_preds):.4f} to {np.max(all_val_preds):.4f}")
    print(
        f"True range: {np.min(all_val_trues):.4f} to {np.max(all_val_trues):.4f}")








    results = calculate_leaderboard_score(all_val_trues, all_val_preds)

    print(f"\nLeaderboard Score Results:")
    print(f"Component A (Normalized RMSE): {results['component_a']:.4f}")
    print(f"Component B (R²): {results['component_b']:.4f}")
    print(f"Final Score: {results['score']:.4f}")
    print(f"Normalized RMSE: {results['normalized_rmse']:.4f}")

    print(f"\nScore Breakdown:")
    print(f"  - A contribution: {0.4 * (1 - results['component_a']):.4f}")
    print(f"  - B contribution: {0.6 * results['component_b']:.4f}")
    print(f"  - Total Score: {results['score']:.4f}")

    print(f"\nPerformance Assessment:")
    if results['score'] >= 0.7:
        print("  - Excellent performance! 🎉")
    elif results['score'] >= 0.6:
        print("  - Good performance! 👍")
    elif results['score'] >= 0.53:
        print("  - Average performance. 📊")
    else:
        print("  - Needs improvement. 📈")













    print("\nGenerating predictions for test data...")

    # 테스트 데이터에 대한 예측
    test_predictions = np.zeros(len(X_test))

    for fold in range(CFG['N_FOLDS']):
        fold_preds = np.zeros(len(X_test))

        for model_type in ['lgb', 'xgb', 'cat']:
            if model_type == 'lgb':
                fold_preds += models[model_type][fold].predict(X_test)
            elif model_type == 'xgb':
                fold_preds += models[model_type][fold].predict(X_test)
            else:
                fold_preds += models[model_type][fold].predict(X_test)

        fold_preds /= 3
        test_predictions += fold_preds

    test_predictions /= CFG['N_FOLDS']

    print(f"Test predictions shape: {test_predictions.shape}")
    print(
        f"Test predictions range: {np.min(test_predictions):.4f} to {np.max(test_predictions):.4f}")

    # 예측 결과 저장
    np.save('test_predictions.npy', test_predictions)
    print("Test predictions saved to test_predictions.npy")

    # 예측 결과를 CSV로도 저장 (필요한 경우)
    test_results_df = pd.DataFrame({
        'predicted_pIC50': test_predictions
    })
    test_results_df.to_csv('test_predictions.csv', index=False)
    print("Test predictions saved to test_predictions.csv")

    print("\nTraining completed!")



