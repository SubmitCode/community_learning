# AUTOGENERATED! DO NOT EDIT! File to edit: 00_data_prep.ipynb (unless otherwise specified).

__all__ = ['download_santander_data', 'unzip', 'SandanderData']

# Cell
import os
import gdown
import pandas as pd
from nbdev import export
from pathlib import Path
from zipfile import ZipFile


# Cell
def download_santander_data():
    """downloads the data from gdrive to the data folder"""
    dest = Path('data/raw/train_ver2.csv.zip')

    #only download if file not already exists
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)

        gdown.download(
            'https://drive.google.com/uc?export=download&id=1H-dFcuqI99OkXAawsTMkmylBz3exNUgZ',
            'data/raw/train_ver2.csv.zip',
            quiet=False)
    else:
        print(f"file {dest} already exists.")

    return dest

# Cell
def unzip(path:[Path,str], dest:[Path,str]='.'):
    """unzips a given file"""
    path = Path(path)
    dest = path.parent / path.stem

    if not dest.exists():
        with ZipFile(str(path), 'r') as zipObj:
            zipObj.extractall(str(path.parent))
            print(f'extracted to {path.parent / path.stem}')
    else:
        print(f"file {dest} already exists!")

    return dest

# Cell
# laden der Daten in einen DataFrame
class SandanderData:
    """class which handles the data in a pipline style"""

    def load_csv(self, path_file:[Path, str], limit_rows=None):
        """load csv file to a pandas df"""

        if limit_rows is None:
            self.df = pd.read_csv(
                path_file,
                dtype={
                    "sexo":str,
                    "ind_nuevo":str,
                    "ult_fec_cli_1t":str,
                    "indext":str})

        self.df = pd.read_csv(
            path_file,
            dtype={
                "sexo":str,
                "ind_nuevo":str,
                "ult_fec_cli_1t":str,
                "indext":str},
            nrows=limit_rows
        )

        return self

    def convert_dates(self):
        """converting the dates of the df"""
        self.df["fecha_dato"] = pd.to_datetime(self.df["fecha_dato"],format="%Y-%m-%d")
        self.df["fecha_alta"] = pd.to_datetime(self.df["fecha_alta"],format="%Y-%m-%d")
        return self


    def add_month_feature(self):
        """add buy month to the features"""
        self.df["month"] = pd.DatetimeIndex(self.df["fecha_dato"]).month
        return self


    def add_age_feature(self):
        """add customer age feature"""
        self.df["age"]   = pd.to_numeric(self.df["age"], errors="coerce")
        return self

    def clean_age(self):
        """clean age with NA small and big ages"""
        self.df.loc[self.df.age < 18,"age"]  = (self.df.loc[(self.df.age >= 18)
                                                       & (self.df.age <= 30),"age"].mean(skipna=True))
        self.df.loc[self.df.age > 100,"age"] = (self.df.loc[(self.df.age >= 30)
                                                       & (self.df.age <= 100),"age"].mean(skipna=True))
        self.df["age"].fillna(self.df["age"].mean(),inplace=True)
        self.df["age"] = self.df["age"].astype(int)
        return self


    def clean_ind_nuevo(self):
        """ind_nuevo indicates a new customer. We replace missing values with one"""
        self.df.loc[self.df["ind_nuevo"].isnull(),"ind_nuevo"] = 1
        return self


    def clean_antiguedad(self):
        """antiguedad means senority. All missing antiguedad have the same NAs as the ind_nuevo."""
        self.df.antiguedad = pd.to_numeric(self.df.antiguedad,errors="coerce")
        self.df.loc[self.df.antiguedad.isnull(),"antiguedad"] = self.df.antiguedad.min()
        self.df.loc[self.df.antiguedad < 0.0, "antiguedad"]  = 0 # Thanks @StephenSmith for bug-find
        return self

    def replace_missing_dates_with_median(self):
        """replace missing fecha_alta with median dates"""
        dates=self.df.loc[:,"fecha_alta"].sort_values().reset_index()
        median_date = int(np.median(dates.index.values))
        self.df.loc[self.df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]
        return self

    def clean_indrel(self):
        """
        indrel value of 1 indicates primary customer 99 means primary customer at the beginning
        of the month but not end of the month"""
        self.df.loc[self.df.indrel.isnull(),"indrel"] = 1
        return self

    def drop_tipodom(self):
        """drop tipodom - Adres type"""
        self.df.drop(["tipodom","cod_prov"],axis=1,inplace=True)
        return self

    def clean_ind_actividad_cliente(self):
        """we replace NANs with the median"""
        self.df.loc[self.df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = \
        self.df["ind_actividad_cliente"].median()
        return self

    def clean_nomprov(self):
        """remove special character and NANs"""
        self.df.loc[self.df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
        self.df.loc[self.df.nomprov.isnull(),"nomprov"] = "UNKNOWN"
        return self


    def clean_renta(self):
        """fill in NaNs with median value from its region"""
        grouped = df.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
        new_incomes = pd.merge(df,grouped,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]
        new_incomes = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")
        df.sort_values("nomprov",inplace=True)
        df = df.reset_index()
        new_incomes = new_incomes.reset_index()

        df.loc[df.renta.isnull(),"renta"] = new_incomes.loc[df.renta.isnull(),"renta"].reset_index()
        df.loc[df.renta.isnull(),"renta"] = df.loc[df.renta.notnull(),"renta"].median()
        df.sort_values(by="fecha_dato",inplace=True)



