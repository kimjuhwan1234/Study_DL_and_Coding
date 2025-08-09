import sys
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys, Lipinski
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import optuna
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# 현재 스크립트의 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

warnings.filterwarnings("ignore")

CFG = {
    "NBITS": 2048,
    "SEED": 42,
    "N_FOLDS": 5,
    "N_TRIALS": 10,  # Optuna 시도 횟수 증가
    "ENSEMBLE_WEIGHTS": [0.4, 0.3, 0.3],  # LGB, XGB, Cat 순서
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(CFG['SEED'])


def IC50_to_pIC50(ic50_nM):
    """IC50 값을 pIC50으로 변환"""
    ic50_nM = np.clip(ic50_nM, 1e-10, None)
    return 9 - np.log10(ic50_nM)


def pIC50_to_IC50(pIC50):
    """pIC50 값을 IC50으로 변환"""
    return 10 ** (9 - pIC50)


def calculate_molecular_descriptors(smiles_list):
    """SMILES 문자열로부터 분자 특성 계산"""
    descriptors = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            desc = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
                'RingCount': Descriptors.RingCount(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'MolarRefractivity': Descriptors.MolMR(mol),
                'LabuteASA': Descriptors.LabuteASA(mol),
            }
            descriptors.append(desc)
        else:
            descriptors.append({})

    return descriptors


def calculate_fingerprints(smiles_list, radius=2, nBits=2048):
    """Morgan fingerprints 및 MACCS keys 계산"""
    fingerprints = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Morgan fingerprints
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=nBits)
            morgan_array = np.array(morgan_fp)

            # MACCS keys
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            maccs_array = np.array(maccs_fp)

            fingerprints.append({
                'morgan': morgan_array,
                'maccs': maccs_array
            })
        else:
            fingerprints.append(
                {'morgan': np.zeros(nBits), 'maccs': np.zeros(167)})

    return fingerprints


def calculate_physicochemical_properties(smiles_list):
    """물리화학적 특성 계산"""
    properties = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            props = {
                # Lipinski 규칙 관련
                'Lipinski_HBD': Descriptors.NumHDonors(mol),
                'Lipinski_HBA': Descriptors.NumHAcceptors(mol),
                'Lipinski_MW': Descriptors.MolWt(mol),
                'Lipinski_LogP': Descriptors.MolLogP(mol),

                # 추가 물리화학적 특성
                'PolarSurfaceArea': Descriptors.TPSA(mol),
                'MolarRefractivity': Descriptors.MolMR(mol),
                'TopologicalSurfaceArea': Descriptors.LabuteASA(mol),

                # 구조적 특성
                'NumAtoms': mol.GetNumAtoms(),
                'NumBonds': mol.GetNumBonds(),
                'NumRings': Descriptors.RingCount(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            }
            properties.append(props)
        else:
            properties.append({})

    return properties


def calculate_complexity_metrics(smiles_list):
    """분자 복잡도 관련 지표 계산"""
    complexity_metrics = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            metrics = {
                'BertzCT': Descriptors.BertzCT(mol),  # Bertz 복잡도 지수
                'Ipc': Descriptors.Ipc(mol),  # 정보 내용
            }
            complexity_metrics.append(metrics)
        else:
            complexity_metrics.append({})

    return complexity_metrics


def calculate_drug_likeness(smiles_list):
    """약물 유사성 특성 계산"""
    drug_properties = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Lipinski 규칙 준수 여부
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            lipinski_violations = sum([
                mw > 500,
                logp > 5,
                hbd > 5,
                hba > 10
            ])

            # Ghose 규칙
            ghose_violations = sum([
                mw < 160 or mw > 480,
                logp < -0.4 or logp > 5.6,
                mol.GetNumAtoms() < 20 or mol.GetNumAtoms() > 70,
                Descriptors.NumRotatableBonds(mol) > 8
            ])

            # Veber 규칙
            veber_violations = sum([
                Descriptors.TPSA(mol) > 140,
                Descriptors.NumRotatableBonds(mol) > 10
            ])

            properties = {
                'Lipinski_Violations': lipinski_violations,
                'Ghose_Violations': ghose_violations,
                'Veber_Violations': veber_violations,
                'Drug_Likeness_Score': max(0, 4 - lipinski_violations),
                'Total_Violations': lipinski_violations + ghose_violations + veber_violations
            }
            drug_properties.append(properties)
        else:
            drug_properties.append({})

    return drug_properties


def calculate_structural_features(smiles_list):
    """분자 구조 기반 특성 계산"""
    structural_features = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            features = {
                # 원자 타입별 개수
                'Num_C': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'C']),
                'Num_N': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'N']),
                'Num_O': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'O']),
                'Num_S': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'S']),
                'Num_F': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'F']),
                'Num_Cl': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'Cl']),
                'Num_Br': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'Br']),
                'Num_I': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'I']),

                # 결합 타입별 개수
                'Num_SingleBonds': len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE]),
                'Num_DoubleBonds': len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]),
                'Num_TripleBonds': len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE]),
                'Num_AromaticBonds': len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC]),

                # 고리 관련 특성 (간단한 버전)
                'NumRings': Descriptors.RingCount(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            }
            structural_features.append(features)
        else:
            structural_features.append({})

    return structural_features


def create_chemical_features_pipeline(smiles_list):
    """모든 화학적 특성을 통합하는 파이프라인"""

    print("Calculating molecular descriptors...")
    molecular_descriptors = calculate_molecular_descriptors(smiles_list)

    print("Calculating fingerprints...")
    fingerprints = calculate_fingerprints(smiles_list)

    print("Calculating physicochemical properties...")
    physicochemical_props = calculate_physicochemical_properties(smiles_list)

    print("Calculating complexity metrics...")
    complexity_metrics = calculate_complexity_metrics(smiles_list)

    print("Calculating drug likeness...")
    drug_likeness = calculate_drug_likeness(smiles_list)

    print("Calculating structural features...")
    structural_features = calculate_structural_features(smiles_list)

    # 모든 특성을 하나의 딕셔너리로 통합
    combined_features = []
    for i in range(len(smiles_list)):
        combined = {}
        combined.update(molecular_descriptors[i])
        combined.update(physicochemical_props[i])
        combined.update(complexity_metrics[i])
        combined.update(drug_likeness[i])
        combined.update(structural_features[i])

        # Fingerprints는 별도로 처리
        if 'morgan' in fingerprints[i]:
            combined['morgan_fp'] = fingerprints[i]['morgan']
        if 'maccs' in fingerprints[i]:
            combined['maccs_fp'] = fingerprints[i]['maccs']

        combined_features.append(combined)

    return combined_features


def select_important_chemical_features(features_dict, target_values, threshold=0.01):
    """중요한 화학적 특성 선택"""

    # 특성을 DataFrame으로 변환
    feature_df = pd.DataFrame(features_dict)

    # 결측값 처리
    feature_df = feature_df.fillna(0)

    # 숫자가 아닌 열 제거
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
    feature_df = feature_df[numeric_columns]

    # 무한값 처리
    feature_df = feature_df.replace([np.inf, -np.inf], 0)

    # 상관관계가 높은 특성 제거
    corr_matrix = feature_df.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(
        upper_tri[column] > 0.95)]
    feature_df = feature_df.drop(columns=to_drop)

    # Mutual Information으로 중요도 계산 (안전한 버전)
    try:
        mi_scores = mutual_info_regression(feature_df, target_values)
        mi_scores_df = pd.DataFrame({
            'feature': feature_df.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        # 임계값 이상의 특성만 선택
        important_features = mi_scores_df[mi_scores_df['mi_score']
                                          > threshold]['feature'].tolist()
    except Exception as e:
        print(f"Mutual information calculation failed: {e}")
        # 실패 시 모든 특성 사용
        important_features = feature_df.columns.tolist()
        mi_scores_df = pd.DataFrame({
            'feature': feature_df.columns,
            'mi_score': [0.1] * len(feature_df.columns)
        })

    return important_features, mi_scores_df


def combine_chemical_with_existing_features(X_existing, chemical_features, important_features=None):
    """기존 특성과 화학적 특성 결합"""

    # 화학적 특성을 DataFrame으로 변환
    chem_df = pd.DataFrame(chemical_features)

    # 중요 특성만 선택 (지정된 경우)
    if important_features:
        chem_df = chem_df[important_features]

    # 결측값 처리
    chem_df = chem_df.fillna(0)

    # 기존 특성과 결합
    X_combined = np.hstack([X_existing, chem_df.values])

    return X_combined, chem_df.columns.tolist()


def load_and_preprocess_data():
    """데이터 로드 및 전처리"""

    print("Loading data...")

    # ChEMBL 데이터 로드
    chembl = pd.read_csv("JUMP_AI/data/open/ChEMBL_ASK1(IC50).csv", sep=';')
    chembl.columns = chembl.columns.str.strip().str.replace('"', '')
    chembl = chembl[chembl['Standard Type'] == 'IC50']
    chembl = chembl[['Smiles', 'Standard Value']].rename(
        columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}).dropna()
    chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')
    chembl['pIC50'] = IC50_to_pIC50(chembl['ic50_nM'])

    # PubChem 데이터 로드
    pubchem = pd.read_csv(
        "JUMP_AI/data/open/Pubchem_ASK1.csv", low_memory=False)
    pubchem = pubchem[['SMILES', 'Activity_Value']].rename(
        columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}).dropna()
    pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')
    pubchem['pIC50'] = IC50_to_pIC50(pubchem['ic50_nM'])

    # 데이터 통합
    total = pd.concat([chembl, pubchem], ignore_index=True)
    total = total.drop_duplicates(subset='smiles')
    total = total[total['ic50_nM'] > 0].dropna()

    print(f"Total data shape: {total.shape}")
    print(
        f"pIC50 range: {total['pIC50'].min():.2f} to {total['pIC50'].max():.2f}")

    return total


def create_enhanced_features(total_data):
    """화학적 특성을 추가한 특성 생성"""

    print("Creating enhanced features...")

    # 기본 fingerprints 생성
    total_data['fingerprint'] = total_data['smiles'].apply(
        lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(x), 2, nBits=CFG['NBITS']))
        if Chem.MolFromSmiles(x) is not None else np.zeros(CFG['NBITS'])
    )

    # 화학적 특성 계산
    chemical_features = create_chemical_features_pipeline(
        total_data['smiles'].tolist())

    # 중요 특성 선택
    important_features, mi_scores = select_important_chemical_features(
        chemical_features, total_data['pIC50'].values, threshold=0.01
    )

    print(f"Selected {len(important_features)} important chemical features")
    print("Top 10 chemical features by importance:")
    print(mi_scores.head(10))

    # 특성 결합
    fp_stack = np.stack(total_data['fingerprint'].values)
    X_enhanced, chem_feature_names = combine_chemical_with_existing_features(
        fp_stack, chemical_features, important_features
    )

    print(f"Enhanced feature matrix shape: {X_enhanced.shape}")

    return X_enhanced, total_data['pIC50'].values, chem_feature_names, mi_scores


def calculate_leaderboard_score(y_true, y_pred):
    """리더보드 점수 계산"""
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


def objective_lgb(trial, X, y, cv_scores, n_folds=5, seed=42):
    """LightGBM 하이퍼파라미터 최적화"""
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
    """XGBoost 하이퍼파라미터 최적화"""
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
    """CatBoost 하이퍼파라미터 최적화"""
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


def main():
    """메인 실행 함수"""

    print("JUMP AI - Enhanced Chemical Features Pipeline")
    print("=" * 60)

    # 데이터 로드 및 전처리
    total_data = load_and_preprocess_data()

    # 화학적 특성 추가
    X_enhanced, y, chem_feature_names, mi_scores = create_enhanced_features(
        total_data)

    # 훈련/검증 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=CFG['SEED']
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # 하이퍼파라미터 최적화
    cv_scores = {"lgb": [], "xgb": [], "cat": []}

    print("\nTuning LightGBM hyperparameters...")
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(
        lambda trial: objective_lgb(
            trial, X_train, y_train, cv_scores, CFG["N_FOLDS"], CFG["SEED"]),
        n_trials=CFG["N_TRIALS"]
    )

    print("\nTuning XGBoost hyperparameters...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(
        lambda trial: objective_xgb(
            trial, X_train, y_train, cv_scores, CFG["N_FOLDS"], CFG["SEED"]),
        n_trials=CFG["N_TRIALS"]
    )

    print("\nTuning CatBoost hyperparameters...")
    study_cat = optuna.create_study(direction="maximize")
    study_cat.optimize(
        lambda trial: objective_cat(
            trial, X_train, y_train, cv_scores, CFG["N_FOLDS"], CFG["SEED"]),
        n_trials=CFG["N_TRIALS"]
    )

    # 최종 모델 훈련
    print("\nTraining final models...")

    # LightGBM
    best_params_lgb = study_lgb.best_params
    best_params_lgb.update({
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "random_state": CFG["SEED"],
        "verbose": -1,
    })

    # XGBoost
    best_params_xgb = study_xgb.best_params
    best_params_xgb.update({
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 1000,
        "random_state": CFG["SEED"],
        "verbosity": 0,
    })

    # CatBoost
    best_params_cat = study_cat.best_params
    best_params_cat.update({
        "loss_function": "RMSE",
        "iterations": 1000,
        "random_state": CFG["SEED"],
        "verbose": False,
    })

    # 모델 훈련
    lgb_model = lgb.LGBMRegressor(**best_params_lgb)
    lgb_model.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(**best_params_xgb)
    xgb_model.fit(X_train, y_train)

    cat_model = cb.CatBoostRegressor(**best_params_cat)
    cat_model.fit(X_train, y_train, verbose=False)

    # 앙상블 예측
    lgb_pred = lgb_model.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)
    cat_pred = cat_model.predict(X_val)

    ensemble_pred = (0.4 * lgb_pred + 0.3 * xgb_pred + 0.3 * cat_pred)

    # 성능 평가
    results = calculate_leaderboard_score(y_val, ensemble_pred)

    print(f"\nFinal Results:")
    print(f"Component A (Normalized RMSE): {results['component_a']:.4f}")
    print(f"Component B (R²): {results['component_b']:.4f}")
    print(f"Final Score: {results['score']:.4f}")

    # 모델 저장
    models = {
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'cat_model': cat_model,
        'best_params_lgb': best_params_lgb,
        'best_params_xgb': best_params_xgb,
        'best_params_cat': best_params_cat,
        'chem_feature_names': chem_feature_names,
        'mi_scores': mi_scores
    }

    with open('data/enhanced_models.pkl', 'wb') as f:
        pickle.dump(models, f)

    print("Enhanced models saved to data/enhanced_models.pkl")

    # 특성 중요도 저장
    feature_importance_df = pd.DataFrame({
        'feature': chem_feature_names,
        'importance': mi_scores[mi_scores['feature'].isin(chem_feature_names)]['mi_score'].values
    }).sort_values('importance', ascending=False)

    feature_importance_df.to_csv(
        'data/chemical_feature_importance.csv', index=False)
    print("Feature importance saved to data/chemical_feature_importance.csv")


if __name__ == "__main__":
    main()
