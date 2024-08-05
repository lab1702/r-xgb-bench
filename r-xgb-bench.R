library(xgboost)
library(microbenchmark)


data(agaricus.train)

dtrain <- xgb.DMatrix(
  agaricus.train$data,
  label = agaricus.train$label
)


microbenchmark(
  default = xgb.cv(
    data = dtrain,
    nfold = 10,
    objective = "binary:logistic",
    nrounds = 100,
    verbose = 0
  ),
  single = xgb.cv(
    data = dtrain,
    nfold = 10,
    objective = "binary:logistic",
    nrounds = 100,
    verbose = 0,
    nthread = 1
  ),
  times = 10
)
