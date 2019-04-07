import pandas as pd
import sys

# train_file = "F:/ml/A3/Q2/data/poker-hand-training-true.data"
# test_file = "F:/ml/A3/Q2/data/poker-hand-testing.data"

# train_save = "F:/ml/A3/Q2/data/poker-hand-training-true-processed.data"
# test_save = "F:/ml/A3/Q2/data/poker-hand-testing-processed.data"
# test_save_small = "F:/ml/A3/Q2/data/poker-hand-testing-processed-small.data"

train_file = sys.argv[1]
test_file = sys.argv[2]

train_save = sys.argv[3]
test_save = sys.argv[4]

test_save_small = test_save + '.small'

dfTr = pd.read_csv(train_file, header=None)
dfTe = pd.read_csv(test_file, header=None)

#################################
# Random UnderSampling ##########

# dfTr_cl = [dfTr[dfTr[10] == i] for i in range(10)]
# dfTr_count = [len(k) for k in dfTr_cl]
# dfTr_sample = [dfTr_cl[i].sample(min(2000, dfTr_count[i])) for i in range(10)]
# dfTr = pd.concat(dfTr_sample)
# print dfTr

#################################
# One hot encoding ##############

X_Tr = dfTr.iloc[:,:10]
X_Te = dfTe.iloc[:,:10]
Y_Tr = dfTr.iloc[:,10:]
Y_Te = dfTe.iloc[:,10:]

X_combined = pd.concat([X_Tr,X_Te],keys=[0,1])
X_combined = pd.get_dummies(X_combined, columns=[0,1,2,3,4,5,6,7,8,9])

Y_combined = pd.concat([Y_Tr,Y_Te],keys=[0,1])
Y_combined = pd.get_dummies(Y_combined, columns=[10])

X_Tr, X_Te = X_combined.xs(0), X_combined.xs(1)
Y_Tr, Y_Te = Y_combined.xs(0), Y_combined.xs(1)

dfTr = pd.concat([X_Tr, Y_Tr], axis=1)
dfTe = pd.concat([X_Te, Y_Te], axis=1)

dfTr.to_csv(train_save, header=False, index=False)
dfTe.to_csv(train_save, header=False, index=False)
dfTe.iloc[:100000,:].to_csv(test_save_small, header=False, index=False)