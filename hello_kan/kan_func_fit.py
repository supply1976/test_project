from kan import KAN, create_dataset, SYMBOLIC_LIB, add_symbolic
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def dbgen():
    f = lambda x, y: np.exp(np.sin(np.pi*x) + y**2)
    xsamp = np.linspace(-1, 1, 50)
    ysamp = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(xsamp, ysamp)
    Z = f(X, Y)
    #plt.pcolormesh(X, Y, Z)
    #
    data = np.stack([X, Y, Z], axis=-1)
    data = data.reshape(-1, 3)
    data_input, data_label = data[:, 0:-1], data[:, -1]
    train_input, test_input, train_label, test_label = train_test_split(
        data_input, data_label, test_size=0.5)
    dataset={}
    dataset['train_input'] = torch.from_numpy(train_input)
    dataset['test_input']  = torch.from_numpy(test_input)
    dataset['train_label'] = torch.from_numpy(train_label.reshape(-1,1))
    dataset['test_label']  = torch.from_numpy(test_label.reshape(-1,1))
    return dataset

def main():
    dataset = dbgen()

    #f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    #dataset = create_dataset(f, n_var=2)

    for k in dataset.keys():
        print(k, dataset[k].shape)

    # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. 
    # cubic spline (k=3), 5 grid intervals (grid=5).
    model = KAN(width=[2,5,5,1], grid=5, k=3, seed=0)
    
    # initialize
    model(dataset['train_input'])
    #model.plot(beta=100)

    # train the model
    model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);
    model.plot()

    # Prune KAN
    model.prune()
    model.plot(mask=True)

    # replot
    model = model.prune()
    model(dataset['train_input'])
    model.plot()

    # continue training
    model.train(dataset, opt="LBFGS", steps=50)
    model.plot()


main()
plt.show()


#model = model.prune()
#model(dataset['train_input'])
#model.plot(beta=20)

