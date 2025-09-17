import pandas as pd
import numpy as np

def transform_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    df_raw: DataFrame with columns 
    ['year','month','day','order','country','session_id','page1_main_category','page2_clothing_model',
    'colour','location','model_photography','price','price_2','page']
    Returns: DataFrame of engineered session-level features (same as session_features.csv except targets)
    """
    # ensure lower-case
    df = df_raw.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # datetime
    df['date'] = pd.to_datetime(dict(year=df['year'],
                                     month=df['month'],
                                     day=df['day']),
                                errors='coerce')
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month_name'] = df['date'].dt.month_name()

    df['purchase_flag'] = np.where(df['order'] > 0, 1, 0)
    df['line_spend'] = df['price'] * df['order']

    session_features = df.groupby('session_id').agg(
        n_unique_categories=('page1_main_category', 'nunique'),
        n_unique_models=('page2_clothing_model', 'nunique'),
        n_clicks=('page', 'count'),
        avg_price_viewed=('price', 'mean'),
        total_price_viewed=('price', 'sum'),
        total_spend=('line_spend', 'sum'),
        purchase_flag=('purchase_flag', 'max'),
        country=('country', lambda x: x.mode()[0] if len(x)>0 else np.nan),
        most_common_category=('page1_main_category', lambda x: x.mode()[0] if len(x)>0 else np.nan),
        day_of_week=('day_of_week', lambda x: x.mode()[0] if len(x)>0 else np.nan),
        month=('month', lambda x: x.mode()[0] if len(x)>0 else np.nan)
    ).reset_index()

    session_features['avg_spend_per_click'] = session_features['total_spend'] / session_features['n_clicks']
    session_features['avg_price_per_click'] = session_features['total_price_viewed'] / session_features['n_clicks']
    session_features = session_features.replace([np.inf,-np.inf],0).fillna(0)

    return session_features
