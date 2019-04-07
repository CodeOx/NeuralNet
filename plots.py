import pandas as pd
import matplotlib.pyplot as plt

hl_units = [5, 10, 15, 20, 25]
df = pd.read_csv('logs_part_e2.csv', header=None)

train_accuracy = df[0].values
test_accuracy = df[1].values
time = df[2].values

l1 = plt.plot(hl_units, train_accuracy,label='Train accuracy')
l2 = plt.plot(hl_units, test_accuracy,label='Test accuracy')
plt.xlabel('hidden layer units')
plt.ylabel('accuracy')
plt.legend(loc='upper left')
plt.savefig('accuracy_single_layer_e2.png')
plt.show()

l1 = plt.plot(hl_units, time,label='Time')
plt.xlabel('hidden layer units')
plt.ylabel('train time (s)')
plt.legend(loc='upper left')
plt.savefig('time_single_layer_e2.png')
plt.show()