# AUTOGENERATED! DO NOT EDIT! File to edit: 04_base_model.ipynb (unless otherwise specified).

__all__ = ['load_data', 'get_shift_cols', 'feature_cols', 'target_cols', 'get_product_dict', 'get_product_reverse_dict',
           'encode_products', 'x_y_split', 'runXGB', 'apk', 'get_results', 'get_base_model_results']

# Cell
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_validate
#from community_learning.features import target_cols
from fastscript import *
from tqdm import tqdm
from itertools import compress

# Cell
def load_data(path_train='data/interim/03_train.csv',
              path_test='data/interim/03_test.csv'):
    """load data"""
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    return (train, test)


# Cell
def get_shift_cols(columns:list):
    """get a list of columns"""
    return [ col for col in columns if col[-2:] == '_s']

# Cell
feature_cols = ['ind_empleado', 'sexo', 'age', 'renta', 'ind_nuevo',
                'indrel', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext',
                'conyuemp', 'indfall', 'tipodom', 'ind_actividad_cliente',
                'segmento', 'antiguedad', 'pais_residencia', 'canal_entrada',
                'ind_cco_fin_ult1_s', 'ind_cder_fin_ult1_s', 'ind_cno_fin_ult1_s',
                'ind_ctju_fin_ult1_s', 'ind_ctma_fin_ult1_s', 'ind_ctop_fin_ult1_s',
                'ind_ctpp_fin_ult1_s', 'ind_deco_fin_ult1_s', 'ind_deme_fin_ult1_s',
                'ind_dela_fin_ult1_s', 'ind_ecue_fin_ult1_s', 'ind_fond_fin_ult1_s',
                'ind_hip_fin_ult1_s', 'ind_plan_fin_ult1_s', 'ind_pres_fin_ult1_s',
                'ind_reca_fin_ult1_s', 'ind_tjcr_fin_ult1_s', 'ind_valo_fin_ult1_s',
                'ind_viv_fin_ult1_s', 'ind_nomina_ult1_s', 'ind_nom_pens_ult1_s',
                'ind_recibo_ult1_s']

target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
               'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
               'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
               'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
               'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
               'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

#feature_cols += get_shift_cols(train.columns) #add the shifted products as feature columns



# Cell
def get_product_dict(df:pd.DataFrame):
    """returns product_name: integer pairs"""
    products = sorted(list(train['y'].unique()))
    return { product : i for i, product in enumerate(products) }

def get_product_reverse_dict(df:pd.DataFrame):
    """returns product_name: integer pairs"""
    products = list(train['y'].unique())
    return { i : product for i, product in enumerate(products) }


# Cell
def encode_products(df:pd.DataFrame):
    """encode products with integer"""
    product_dict = get_product_dict(df)
    df['y'] = df['y'].map(lambda x: product_dict[x]).astype(np.int8)
    return df

# Cell
def x_y_split(df:pd.DataFrame):
    """returns 2 dataframes for X and Y variables"""
    X = df.drop('y', axis=1)
    y = df['y']
    return (X, y)

# Cell
def runXGB(train_X, train_y, feature_cols, seed_val=0):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05
    param['max_depth'] = 8
    param['silent'] = 1
    param['num_class'] = 22
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = 50

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X[feature_cols], label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)
    return model


# Cell
def apk(actual, predicted, k=7):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


# Cell
def get_results(test_data:pd.DataFrame, model:xgb.core.Booster,  target_cols:list=target_cols):
    """"""
    xgtest = xgb.DMatrix(test[feature_cols])
    preds = model.predict(xgtest)
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)[:,:7]
    preds = pd.DataFrame(preds)
    preds = preds.applymap(lambda x: product_reverse_dict[x])
    preds['added_products'] = preds.apply(lambda x: list(x.values), axis=1)
    preds = preds['added_products']


    test_data['added_products'] = preds
    test_data['truth_list'] = test_data[target_cols].apply(lambda x: list(compress(target_cols, x.values)), axis=1)
    test_data[['added_products', 'truth_list']]
    test['apk'] = test.apply(lambda x: apk(x['truth_list'], x['added_products']),axis=1)
    print(f"mean average precision = {test['apk'].mean()}")
    return test_data[['id', 'added_products', 'truth_list', 'apk']]

# Cell
@call_parse
def get_base_model_results(source_train:Param("source csv file for train", str)='data/interim/03_train.csv',
                           source_test:Param("source csv file for test", str)='data/interim/03_test.csv',
                           dest:Param("destination csv file for the results", str)='data/results/base_model.csv',
                           feature_cols:Param("list of features to use for training", str)=feature_cols):
    """"""
    train_org, test = load_data(source_train, source_test)

    product_dict = get_product_dict(train_org)
    product_reverse_dict = get_product_reverse_dict(train_org)

    train = encode_products(train)

    train_X, train_y = x_y_split(train)

    model = runXGB(train_X, train_y, feature_cols)

    results = get_results(test, model)

    results.to_csv(dest, index=False)

    return results
