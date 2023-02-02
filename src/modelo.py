import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# modelo random forest
def modelo_random_forest(train,max_leaf=[250]):
    y = train['SalePrice']
    X = train.drop(['SalePrice'], axis=1)

    for node in max_leaf:
        model = RandomForestRegressor(max_leaf_nodes=node,)
        model.fit(X, y)
        score = cross_val_score(model, X, y, cv=10)
    return model

#predicción de modelo

def prediccion(test):
    price = model.predict(test_data)
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": price
    })
    return submission

    