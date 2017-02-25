
import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet

from cs231n.data_utils import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

best_net = None  # store the best model into this

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
hidden_size = [x * 50 for x in range(1, 3, 1)]
learning_rate = [5e-5, 1e-4, 5e-4]#, 1e-3, 5e-3, 2e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
training_epoch = [x * 200 for x in range(1, 3, 1)]
regularization_strengths = [0.001, 0.002, 0.05]#, 0.07, 0.1, 0.2, 0.5, 1, 5, 10, 20]
batch_size_choice = [20, 50, 100]#, 150, 200, 350, 400, 500, 600, 700, 1000]

input_size = 32 * 32 * 3
num_classes = 10
learning_rate_decay = 0.95

best_val = -1
best_stats = None
best_hype = {}

whole_time = len(hidden_size) * len(learning_rate) * len(training_epoch) * len(regularization_strengths) * len(
    batch_size_choice)

for bat_size in batch_size_choice:
    for reg_str in regularization_strengths:
        for train_circle in training_epoch:
            for lr_rate in learning_rate:
                for hid_size in hidden_size:
                    whole_time -= 1
                    print "now rest times is:", whole_time
                    """
                    print
                    print "now rest times is:", whole_time
                    print "now hidden size is: ", hid_size
                    print "now learning rate is: ", lr_rate
                    print "now training epoch is:", train_circle
                    print "now regualrization streagths is: ", reg_str
                    print "now batch size is: ", bat_size
                    """
                    net = TwoLayerNet(input_size, hid_size, num_classes)

                    now_stats = net.train(X_train, y_train, X_val, y_val,
                                          learning_rate=lr_rate, learning_rate_decay=0.95,
                                          reg=reg_str, num_iters=train_circle, batch_size=bat_size, verbose=False)

                    y_pred_val = net.predict(X_val)
                    acc_val = np.mean(y_pred_val == y_val)

                    filename = './files/loop_%d_acc_%.2f.txt' % (whole_time,acc_val)

                    content = "now accuracy is:"+str(acc_val)+"now rest times is:" + str(whole_time) +'\n' + "now hidden size is: "+ str(hid_size)+'\n'+"now learning rate is: "+ str(lr_rate)+'\n'+"now training epoch is:"+ str(train_circle)+'\n'+"now regualrization streagths is: "+str(reg_str)+'\n'+"now batch size is: "+str(bat_size)

                    test_file = open(filename, 'w')

                    test_file.write(content)

                    test_file.close()

                    if acc_val > best_val:
                        best_val = acc_val
                        best_stats = now_stats
                        best_net = net
                        best_hype = {
                            'bat': bat_size,
                            'reg': reg_str,
                            'cir': train_circle,
                            'ler': lr_rate,
                            'hid': hid_size,
                            'loop': whole_time
                        }
                        """
                        print "=============================================="
                        print "best accuray is:", best_val
                        print "best hidden size is: ", best_hype['hid']
                        print "best learning rate is: ", best_hype['ler']
                        print "best training epoch is:", best_hype['cir']
                        print "best regualrization streagths is: ", best_hype['reg']
                        print "best batch size is: ", best_hype['bat']
                        print "=============================================="
                        """
                        filename = './files/loop_%d_!!!!!!BEST!!!!!!_acc_%.2f.txt' % (best_hype['loop'], best_val)

                        content = "best accuray is:"+str(best_val)+"best rest times is:" + str(whole_time) + '\n' + "best hidden size is: " + str(best_hype['hid']) + '\n' + "best learning rate is: " + str(best_hype['ler']) + '\n' + "best training epoch is:" + str(best_hype['cir']) + '\n' + "best regualrization streagths is: " + str(best_hype['reg']) + '\n' + "best batch size is: " + str(best_hype['bat'])

                        test_file = open(filename, 'w')

                        test_file.write(content)

                        test_file.close()
#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################

print "=============================================="
print "best accuray is:", best_val
print "best hidden size is: ", best_hype['hid']
print "best learning rate is: ", best_hype['ler']
print "best training epoch is:", best_hype['cir']
print "best regualrization streagths is: ", best_hype['reg']
print "best batch size is: ", best_hype['bat']
print "=============================================="
print

filename = './files/FINAL_BEST_loop_%d_!!!!!!BEST!!!!!!_acc_%.2f.txt' % (best_hype['loop'], best_val)

content = "best accuray is:"+str(best_val)+'\n' + "best hidden size is: " + str(best_hype['hid']) + '\n' + "best learning rate is: " + str(best_hype['ler']) + '\n' + "best training epoch is:" + str(best_hype['cir']) + '\n' + "best regualrization streagths is: " + str(best_hype['reg']) + '\n' + "best batch size is: " + str(best_hype['bat'])

test_file = open(filename, 'w')

test_file.write(content)

test_file.close()