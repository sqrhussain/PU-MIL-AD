from PUMA import PUMA


def run(x_train, y_train, x_test, seed):
    # Hyperparameters
    n_pos = 50 #number of positive bags
    n_neg = -1 #number of reliable negative bags: -1 means the same as n_pos

    #Network structure:
    hidden_neurons = [4,2]
    batch_size = 16
    lr = 0.005
    epochs = 300

    puma = PUMA(hidden_neurons = hidden_neurons, learning_rate=lr, epochs = epochs, batch_size = batch_size,
                random_state = seed, n_neg = n_neg, verbose = True)

    puma.fit(x_train, y_train)

    bag_pi, instancepij = puma.decision_function(X_test)
    bag_pi = bag_pi.reshape(-1)
    instancepij = instancepij.reshape(-1)

    return bag_pi, instancepij
