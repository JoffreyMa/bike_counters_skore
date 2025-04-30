from pathlib import Path
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import SplineTransformer
import numpy as np

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )

def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    # Convert the date column to datetime
    df_ext["date"] = df_ext["date"].astype('datetime64[ms]')

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    # Convert the date column to datetime
    X["date"] = X["date"].astype('datetime64[ms]')
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t", "rr1", "rr3", "rr6",  "u"]].sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore", sparse_output=False), date_cols),
            ("cat", categorical_encoder, categorical_cols),
            ("cyclic_month", periodic_spline_transformer(12, n_splines=6), ["month"]),
            ("cyclic_weekday", periodic_spline_transformer(7, n_splines=3), ["weekday"]),
            ("cyclic_hour", periodic_spline_transformer(24, n_splines=12), ["hour"]),
        ],
    )
    regressor = HistGradientBoostingRegressor(max_iter=1000, min_samples_leaf=200, l2_regularization=0.1)

    pipe = make_pipeline(FunctionTransformer(_merge_external_data, validate=False), 
                         date_encoder, 
                         preprocessor, 
                         regressor)

    return pipe
