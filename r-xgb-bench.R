
library(xgboost)
library(microbenchmark)


data(agaricus.train)

dtrain <- xgb.DMatrix(
    agaricus.train$data,
    label=agaricus.train$label
)


bm <- microbenchmark(
    nthreadDefault=xgb.cv(
        data=dtrain,
        nfold=10,
        objective="binary:logistic",
        nrounds=200,
        verbose=0
    ),
    nthread1=xgb.cv(
        data=dtrain,
        nfold=10,
        objective="binary:logistic",
        nrounds=200,
        verbose=0,
        nthread=1
    ),
    times=10
)

print(bm)
