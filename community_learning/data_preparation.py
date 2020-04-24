# AUTOGENERATED! DO NOT EDIT! File to edit: 02_data_Cleaning.ipynb (unless otherwise specified).

__all__ = ['load_data', 'clean_age', 'clean_renta', 'clean_all_data']

# Cell
import pandas as pd
import numpy as np

from fastscript import *

# Cell
def load_data(path='data/interim/01_train.csv'):
    """load data"""
    return pd.read_csv(path)

# Cell
def clean_age(df:pd.DataFrame):
    """clean age"""
    df.loc[df.age < 18,"age"]  = (df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True))
    df.loc[df.age > 100,"age"] = (df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True))
    df["age"].fillna(df["age"].mean(),inplace=True)
    df["age"] = df["age"].astype(int)
    return df

# Cell
def clean_renta(df:pd.DataFrame):
    """fill in NaNs with median value from its region"""
    df = df[df["renta"] > 0].copy()
    grouped = df.groupby("cod_prov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
    new_incomes = pd.merge(df,grouped,how="inner",on="cod_prov").loc[:, ["cod_prov","renta_y"]]
    new_incomes = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("cod_prov")
    df.sort_values("cod_prov",inplace=True)
    df = df.reset_index()
    new_incomes = new_incomes.reset_index()

    df.loc[df.renta < 0,"renta"] = new_incomes.loc[df.renta.isnull(),"renta"].reset_index()
    df.loc[df.renta < 0,"renta"] = df.loc[df.renta.notnull(),"renta"].median()
    df['renta'] = df['renta'].astype(np.int32)
    df.sort_values(by="month_int",inplace=True)
    df.drop('index', inplace=True, axis=1)
    return df


# Cell
@call_parse
def clean_all_data(source:Param("source csv file", str) = 'data/interim/01_train.csv',
                   dest:Param("destination csv file", str) = 'data/interim/02_train.csv'):
    """main function to clean all data"""
    data = load_data(source)
    data = clean_age(data)
    data = clean_renta(data)
    data.to_csv(dest, index=False)
    return data