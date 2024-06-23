import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from tqdm.keras import TqdmCallback


def func_dataset(f, samples=2000, input_range=(-1,1)):
    input_arr = np.random.rand(samples, 2)
    input_arr = input_arr*(input_range[1]-input_range[0]) + input_range[0]
    label_arr = np.array([f(a,b) for (a,b) in input_arr])
    label_arr = label_arr.reshape([-1,1])
    input_tensor = tf.convert_to_tensor(input_arr, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(label_arr, dtype=tf.float32)
    dataset = (input_tensor, label_tensor)
    return dataset


def build_MLPmodel(units=[5, 1]):
    x = keras.Input(shape=(2,))
    outputs=[x]
    for i, u in enumerate(units):
        actf = None if i==len(units)-1 else 'relu'
        _h = keras.layers.Dense(units=u, activation=actf)
        y = _h(outputs[-1])
        outputs.append(y)
    model = keras.Model(inputs=x, outputs=outputs[-1])
    model.summary()
    return model


def build_KANmodel(units=[5, 1], grid_size=5, spline_order=3):
    x = keras.Input(shape=(2,))
    outputs=[x]
    for i, u in enumerate(units):
        _h = DenseKAN(units=u, grid_size=grid_size, spline_order=spline_order)
        y = _h(outputs[-1])
        outputs.append(y)
    model = keras.Model(inputs=x, outputs=outputs[-1])
    model.summary()
    return model


def calc_spline_values(x, grid, spline_order):
    """
    from https://github.com/ZPZhou-lab/tfkan/blob/main/tfkan/ops/spline.py
    
    Calculate B-spline values for the input tensor.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor with shape (batch_size, in_size).
    grid : tf.Tensor
        The grid tensor with shape (in_size, grid_size + 2 * spline_order + 1).
    spline_order : int
        The spline order.

    Returns: tf.Tensor
        B-spline bases tensor of shape (batch_size, in_size, grid_size + spline_order).
        B_ik === B_k(x_i) 
    """
    assert len(tf.shape(x)) == 2
    
    # add a extra dimension to do broadcasting with shape (batch_size, in_size, 1)
    x = tf.expand_dims(x, axis=-1)

    # init the order-0 B-spline bases
    bases = tf.logical_and(
        tf.greater_equal(x, grid[:, :-1]), tf.less(x, grid[:, 1:]))
    bases = tf.cast(bases, x.dtype)
    
    # iter to calculate the B-spline values
    for k in range(1, spline_order + 1):
        bases = (
            (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            * bases[:, :, :-1]
        ) + (
            (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)])
            * bases[:, :, 1:]
        )
    return bases


class DenseKAN(keras.layers.Layer):
    def __init__(self, units, grid_size=5, spline_order=3,
                 grid_range=(-1.0,1.0), **kwargs):
        super(DenseKAN, self).__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        
    def build(self, input_shape):
        in_size = input_shape[-1]
        self.in_size = in_size
        self.spline_basis_size = self.grid_size + self.spline_order
        bound = self.grid_range[1] - self.grid_range[0]

        # build grid
        self.grid = tf.linspace(
            self.grid_range[0] - self.spline_order * bound / self.grid_size,
            self.grid_range[1] + self.spline_order * bound / self.grid_size,
            self.grid_size + 2*self.spline_order +1)

        self.grid = tf.repeat(self.grid[None, :], in_size, axis=0)
        self.grid = tf.Variable(
            initial_value=tf.cast(self.grid, dtype=tf.float32),
            trainable=False,
            dtype=tf.float32,
            name="spline_grd")

        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=(self.in_size, self.spline_basis_size, self.units),
            initializer=tf.keras.initializers.RandomNormal(0.1),
            trainable=True,
            dtype=tf.float32)

    def call(self, inputs, *args, **kwargs):
        Bik = calc_spline_values(inputs, self.grid, self.spline_order)
        spline_out = tf.einsum("bik,iko->bo", Bik, self.spline_kernel)
        return spline_out

    def get_config(self):
        config = super(DenseKAN, self).get_config()
        config.update({
            "units": self.units,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order, 
            "grid_range": self.grid_range})


def model_qor_plot(f, model, model_name):
    xsamp = np.linspace(-1, 1, 50)
    ysamp = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(xsamp, ysamp)
    true_Z = f(X, Y)
    test_samples = np.stack([X, Y], axis=-1).reshape([-1,2])
    pred_Z = model.predict(test_samples).reshape([50,50])
    #
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    axes = axes.flatten()
    axes[0].pcolormesh(X, Y, true_Z)
    axes[1].pcolormesh(X, Y, pred_Z)
    axes[0].set_title("ground truth")
    axes[1].set_title(model_name)


def main():
    # gen training data pairs: (x_i,y_i) --> f(x_i,y_i)
    f = lambda x, y: np.exp(np.sin(np.pi*x) + y**2)
    (ds_input, ds_target) = func_dataset(f)
    
    # MLP model 
    MLPmodel = build_MLPmodel(units=[5, 1])
    MLPmodel.compile(loss='mae', optimizer='adam')
    
    # KAN model
    KANmodel = build_KANmodel(units=[5,1])
    KANmodel.compile(loss='mae', optimizer='adam')

    MLPmodel.fit(x=ds_input, y=ds_target, validation_split=0.2, epochs=100)
    KANmodel.fit(x=ds_input, y=ds_target, validation_split=0.2, epochs=100)

    # check model accuracy
    model_qor_plot(f, model=MLPmodel, model_name="MLP model")
    model_qor_plot(f, model=KANmodel, model_name="KAN model")

main()
plt.show()
