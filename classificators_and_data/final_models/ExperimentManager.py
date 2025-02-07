from tqdm import tqdm
from .LieModel import LieModel
from enum import Enum
from .config import config
import sys
sys.path.append("..")
from data_extractor.data_extractor import load_df, extract_X_y_from_df
import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
from sklearn.model_selection import train_test_split
import logging
import random
from typing import Literal 

logging.basicConfig(level=logging.INFO)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
RESULT_PATH = os.path.abspath(os.path.join(BASE_DIR, 'results'))

config["DATA_PATH"] = DATA_PATH
config["RESULT_PATH"] = RESULT_PATH


class SplitStrategy(Enum):
    RANDOM42 = "Random42"
    RANDOM2137 = "Random2137"
    SMALLTEST42 = "SmallTest42"
    SUBJECTBASED42 = "SubjectBased42"

class ExperimentManager():
    
    def __init__(self, models: list[LieModel], split_strategies: list[SplitStrategy]) -> None:
        self._models = models
        self._split_strategies = split_strategies
        self._df: pd.DataFrame = None        

    def run(self):
        for model in tqdm(self._models, desc="Models Progress"):
            for split_strategy in self._split_strategies:
                metadata = model.determinate_run_info()
                logging.info(f"Running model {metadata['person']}/{metadata['model']} for split strategy {split_strategy.value} ")
                result_exists = self._check_result_exists(model, split_strategy)
                if not result_exists:  
                    self._run_single(model, split_strategy)
                else:
                    logging.info(f"Already calculated. Skipping...\n") 
            self._df = None

    def _check_result_exists(self, model: LieModel, split_strategy: SplitStrategy):
        metadata = model.determinate_run_info()
        path = os.path.join(config["RESULT_PATH"], split_strategy.value, metadata["person"], metadata["model"])
        path = f'{path}.json'
        return os.path.exists(path)

    def _run_single(self, model: LieModel, split_strategy: SplitStrategy):
        preprocess_config = model.determinate_preprocess_config()
        logging.info(f"Getting data...")  
        X_train, X_test, y_train, y_test = self._get_data(split_strategy, preprocess_config)
        logging.info(f"Training...")  
        model.train(X_train, y_train)
        logging.info(f"Predicting...")  
        y_pred = model.predict(X_test)
        logging.info(f"Calculating metrics...")
        metrics = self._calculate_metrics(y_test, y_pred)
        metadata = model.determinate_run_info()
        metadata["split_strategy"] = split_strategy.value
        logging.info(f"Saving...\n")
        self._save_metrics(metrics, metadata)

    def _get_data(self, split_strategy: SplitStrategy, preprocess_config: dict[any, any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._df is not None:
            df = self._df
            logging.info(f"Using buffered data")
        else:
            if isinstance(preprocess_config, list):
                dfs = []
                for pre_config in preprocess_config:
                    df = load_df(config["DATA_PATH"], **pre_config)
                    df = df.query("desired_answer == answer and data_type in ['REAL', 'FAKE']")
                    # df['label'] = df.apply(lambda x: 1 if x.block_no in [1,3] else 0, axis = 1)
                    df.loc[:, 'label'] = df.apply(lambda x: 1 if x.block_no in [1, 4] else 0, axis=1)
                    dfs.append(df)
                Xs_train, Xs_test = [], []
                for df in dfs:
                    df_train, df_test = self._split_df(df, split_strategy)
                    X_train, y_train = extract_X_y_from_df(df_train)
                    X_test, y_test = extract_X_y_from_df(df_test)
                    Xs_train.append(X_train)
                    Xs_test.append(X_test)
                return np.array(Xs_train), np.array(Xs_test), y_train, y_test
            else:
                df = load_df(config["DATA_PATH"], **preprocess_config)
                df = df.query("desired_answer == answer and data_type in ['REAL', 'FAKE']")
                # df['label'] = df.apply(lambda x: 1 if x.block_no in [1,3] else 0, axis = 1)
                df.loc[:, 'label'] = df.apply(lambda x: 1 if x.block_no in [1, 4] else 0, axis=1)
                self._df = df
        df_train, df_test = self._split_df(df, split_strategy)
        X_train, y_train = extract_X_y_from_df(df_train)
        X_test, y_test = extract_X_y_from_df(df_test)
        return X_train, X_test, y_train, y_test
    
    def _split_df(self, df: pd.DataFrame, split_strategy: SplitStrategy) -> tuple[pd.DataFrame, pd.DataFrame]:
        if split_strategy.value == SplitStrategy.RANDOM42.value:
            return self._split_random42(df)
        elif split_strategy.value == SplitStrategy.RANDOM2137.value:
            return self._split_random2137(df)
        elif split_strategy.value == SplitStrategy.SMALLTEST42.value:
            return self._split_smalltest42(df)
        elif split_strategy.value == SplitStrategy.SUBJECTBASED42.value:
            return self._split_subjectbased42(df)
        else:
            raise ValueError(f"Invalid Split Strategy: {split_strategy}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[any, any]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred)
        }

    def _save_metrics(self, metrics: dict[any, any], metadata: dict[any, any]):
        metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
        path = config["RESULT_PATH"]
        self._create_if_not_exist(path)
        path = os.path.join(path, metadata["split_strategy"])
        self._create_if_not_exist(path)
        path = os.path.join(path, metadata["person"])
        self._create_if_not_exist(path)
        path = os.path.join(path, metadata["model"])
        path = f'{path}.json'
        metadata["timestamp"] = str(datetime.now())
        metrics["metadata"]  = metadata
        with open(path, "w", encoding = "utf-8") as file:
            json.dump(metrics, file)
        
    def _create_if_not_exist(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    def _split_random42(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42, stratify = df['label'])
        return df_train, df_test
    
    def _split_random2137(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 2137, stratify = df['label'])
        return df_train, df_test
    
    def _split_smalltest42(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_train, df_test = train_test_split(df, test_size = 0.05, random_state = 42, stratify = df['label'])
        return df_train, df_test
    
    def _split_subjectbased42(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        subjects = df['subject'].unique()
        random.seed(42)
        random.shuffle(subjects)
        split_index = int(len(subjects) * 0.8)
        subjects_train = subjects[:split_index]
        subjects_test = subjects[split_index:]
        df_train = df[df['subject'].isin(subjects_train)]
        df_test = df[df['subject'].isin(subjects_test)]
        return df_train, df_test

        

