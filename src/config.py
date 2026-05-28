# 绝对路径，确保不出错
DATA_DIR = 'data'

# 写代码调试时设为 True，正式跑设为 False
DEBUG_MODE = False

if DEBUG_MODE:
    TRANS_FILE = 'transactions_sample.parquet'
    CUST_FILE = 'customers_sample.parquet'
else:
    TRANS_FILE = 'transactions_train.parquet'
    CUST_FILE = 'customers.parquet'


# Radek 官方原版的超参数
LGBM_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'boosting_type': 'dart',
    'n_estimators': 100,
    'importance_type': 'gain'
}


