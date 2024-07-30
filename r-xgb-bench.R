
library(xgboost)
library(microbenchmark)


data(agaricus.train)

dtrain <- xgb.DMatrix(
    agaricus.train$data,
    label=agaricus.train$label
)


microbenchmark(
    no_nthread=xgb.cv(
        data=dtrain,
        nfold=10,
        objective="binary:logistic",
        nrounds=100,
        verbose=0
    ),
    nthread_48=xgb.cv(
        data=dtrain,
        nfold=10,
        objective="binary:logistic",
        nrounds=100,
        verbose=0,
        nthread=48
    ),
    nthread_01=xgb.cv(
        data=dtrain,
        nfold=10,
        objective="binary:logistic",
        nrounds=100,
        verbose=0,
        nthread=1
    ),
    times=10
)
