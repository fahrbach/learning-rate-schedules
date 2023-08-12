import numpy as np
import scvelo as scv
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

SEED = 0
EPS = 0.1
MULTICLASS = True

# Distribution:
# - Alpha: (481, 10)
# - Beta: (591, 10)
# - Delta: (70, 10)
# - Ductal: (916, 10)
def perturb_and_pick(adata, cell_type):
    assert cell_type in ['Alpha', 'Beta', 'Delta', 'Ductal']
    sample_shape = adata[adata.obs['clusters'] == cell_type].X.shape
    cell_index = np.random.choice(np.arange(sample_shape[0]), size=1, replace=False)
    sample = adata[adata.obs['clusters'] == cell_type].X[cell_index,:]
    # Convert from sparse SparseCSRView to np.ndarray
    sample = np.squeeze(sample.toarray())
    noise_vec = np.random.uniform(low=(1.0 - EPS), high=(1.0 + EPS), size=sample_shape[1])
    return np.multiply(sample, noise_vec)

# TODO(fahrbach): Generalize mixture of probability distributions.
def generate_samples(adata, num_samples, multiclass, p1, p2, p3):
    X = np.zeros((num_samples, adata.X.shape[1]))
    if multiclass:
        Y = np.zeros((num_samples, 4))
    else:
        Y = np.zeros((num_samples, 1))
    for i in range(num_samples):
        r = np.random.randint(1, 10 + 1)
        if r <= p1:
            X[i] = perturb_and_pick(adata, 'Alpha')
            if multiclass:
                Y[i][0] = 1
            else:
                Y[i] = 0.75
        elif r <= p1 + p2:
            X[i] = perturb_and_pick(adata, 'Beta')
            if multiclass:
                Y[i][1] = 1
            else:
                Y[i] = 1.0
        elif r <= p1 + p2 + p3:
            X[i] = perturb_and_pick(adata, 'Delta')
            if multiclass:
                Y[i][2] = 1
            else:
                Y[i] = 0.5
        else:
            X[i] = perturb_and_pick(adata, 'Ductal')
            if multiclass:
                Y[i][3] = 1
            else:
                Y[i] = 0.0
    return X, Y

def create_training_data(adata, num_samples, multiclass, distribution_type):
    if distribution_type == 1:
        return generate_samples(adata, num_samples, multiclass, 0, 0, 1)
    if distribution_type == 2:
        return generate_samples(adata, num_samples, multiclass, 1, 0, 2)
    if distribution_type == 3:
        return generate_samples(adata, num_samples, multiclass, 3, 5, 1)
    return generate_samples(adata, num_samples, multiclass, 0, 0, 0)

def unpack_optimizer_str(optimizer_str):
    tokens = optimizer_str.split(':')
    assert len(tokens) >= 1
    name = tokens[0]
    assert name in ['sgd', 'adam', 'adam-restart']
    if name == 'sgd':
        learning_rate = 0.01
        if len(tokens) >= 2:
            learning_rate = eval(tokens[1])
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif name in ['adam', 'adam-restart']:
        learning_rate = 0.001
        if len(tokens) >= 2:
            learning_rate = eval(tokens[1])
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    return optimizer

# TODO(fahrbach): Use CategoricalCrossentropy if y.shape[1] >= 2.
# (32, 8, 4): Adam --> 400 regret
# (64, 16, 4): Adam --> 250 regret
# (64, 32, 16, 8, 4): Adam --> 200 regret
def build_compiled_model(optimizer_str, multiclass):
    initializer = keras.initializers.GlorotUniform(seed=SEED)
    regularizer = keras.regularizers.L2(1e-4)
    optimizer = unpack_optimizer_str(optimizer_str)
    inputs = keras.Input(shape=(10,), name='input')
    
    if multiclass:
        x_1 = keras.layers.Dense(units=64, activation='elu', name='dense_1',
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer)(inputs)
        x_2 = keras.layers.Dense(units=32, activation='elu', name='dense_2',
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer)(x_1)
        x_3 = keras.layers.Dense(units=16, activation='elu', name='dense_3',
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer)(x_2)
        x_4 = keras.layers.Dense(units=8, activation='elu', name='dense_4',
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer)(x_3)
        outputs = keras.layers.Dense(units=4, kernel_initializer=initializer,
                                     kernel_regularizer=regularizer)(x_4)
        loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    else:
        x = keras.layers.Dense(units=10, activation='elu', name='dense_1', 
                               kernel_initializer=initializer,
                               kernel_regularizer=regularizer)(inputs)
        outputs = keras.layers.Dense(units=1,
                                     activation='sigmoid',
                                     name='prediction',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer)(x)
        loss=keras.losses.MeanSquaredError()

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return model

# TODO(fahrbach): Start caching results to disk?
def run_baseline_experiment_v2(adata, multiclass, num_steps_per_phase, batch_size, optimizer_str):
    keras.backend.clear_session()
    model = build_compiled_model(optimizer_str, multiclass)
    losses = []
    val_losses = []

    distribution_types = [0, 1, 2, 2, 3, 3]
    for phase in range(len(distribution_types)):
        distribution_type = distribution_types[phase]
        if 'restart' in optimizer_str:
            if phase == 0 or distribution_type == distribution_types[phase - 1]:
                if multiclass:
                    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
                else:
                    loss = keras.losses.MeanSquaredError()
                optimizer = unpack_optimizer_str(optimizer_str)
                model.compile(optimizer=optimizer, loss=loss)
        for step in range(num_steps_per_phase):
            lr = keras.backend.eval(model.optimizer._learning_rate)
            print('optimizer_str:', optimizer_str, 'phase:', phase, 'step:', step, 'learning_rate:', lr)
            num_samples = 2 * batch_size
            X, y = create_training_data(adata, num_samples, multiclass, distribution_type)
            history = model.fit(X, y, batch_size=batch_size, epochs=1, validation_split=0.5, shuffle=False)
            # Mean loss and validation loss for epoch
            losses.append(history.history['loss'][0])
            val_losses.append(history.history['val_loss'][0])

    losses = np.array(losses)
    val_losses = np.array(val_losses)
    return losses, val_losses

def plot_velocity_graph(adata):
    # https://scvelo.readthedocs.io/en/stable/VelocityBasics/
    scv.settings.presenter_view = True  # set max width size for presenter view
    scv.set_figure_params('scvelo')  # for beautified visualization
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    #scv.pl.velocity_embedding_stream(adata, basis='umap')
    #scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120)
    scv.pl.velocity_graph(adata, threshold=0.1)

def main():
    np.random.seed(SEED)

    adata = scv.datasets.pancreas()
    print('X.shape (before):', adata.X.shape)

    scv.pp.filter_genes(adata, min_shared_counts=30000)
    scv.pp.normalize_per_cell(adata)
    print('X.shape (after):', adata.X.shape)

    #plot_velocity_graph()

    #X, y = create_training_data(adata, 100, MULTICLASS, 3)
    #print('X.shape', X.shape)
    #print('y.shape', y.shape)
    #return
    #optimizers = ['sgd:0.0001', 'sgd:0.001', 'sgd:0.01', 'sgd:0.1', 'adam', 'adam:restart']
    #optimizers = ['sgd:0.01', 'adam:0.001', 'adam-restart:0.001', 'adam:0.01', 'adam-restart:0.01']
    optimizers = []
    #optimizers += ['adam:0.003', 'adam-restart:0.003']
    optimizers += ['adam:0.01', 'adam-restart:0.01']
    optimizers += ['adam:0.03', 'adam-restart:0.03']
    all_losses = []
    all_val_losses = []
    for optimizer in optimizers:
        NUM_STEPS_PER_PHASE = 100
        BATCH_SIZE = 64
        #NUM_STEPS_PER_PHASE = 100
        #BATCH_SIZE = 64
        losses, val_losses = run_baseline_experiment_v2(adata, MULTICLASS, NUM_STEPS_PER_PHASE, BATCH_SIZE, optimizer)
        all_losses.append(losses)
        all_val_losses.append(val_losses)

    name = 'n' + str(NUM_STEPS_PER_PHASE) + '_'
    name += 'b' + str(BATCH_SIZE) + '_'
    if MULTICLASS:
        name += 'ce'  # cross entropy
    else:
        name += 'ls'  # least squares
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    """
    # Plot loss (train)
    for i in range(len(ETA_LIST)):
        plt.plot(all_losses[i], label='eta:'+str(ETA_LIST[i]))
    plt.xlabel('step')
    plt.ylabel('loss (train)')
    plt.grid()
    plt.legend()
    plt.title('figures/' + name + '_loss_train.png')
    plt.savefig('figures/' + name + '_loss_train.png')
    plt.show()

    # Plot loss (valid)
    for i in range(len(ETA_LIST)):
        plt.plot(all_val_losses[i], label='eta:'+str(ETA_LIST[i]))
    plt.xlabel('step')
    plt.ylabel('loss (valid)')
    plt.grid()
    plt.legend()
    plt.title('figures/' + name + '_loss_valid.png')
    plt.savefig('figures/' + name + '_loss_valid.png')
    plt.show()

    # Plot loss (both)
    for i in range(len(ETA_LIST)):
        plt.plot(all_losses[i], label='eta:'+str(ETA_LIST[i]), c=colors[i])
        plt.plot(all_val_losses[i], label='eta:'+str(ETA_LIST[i]), linestyle='--', c=colors[i])
    plt.xlabel('step')
    plt.ylabel('loss (both)')
    plt.grid()
    plt.legend()
    plt.title('figures/' + name + '_loss_both.png')
    plt.savefig('figures/' + name + '_loss_both.png')
    plt.show()
    """

    # Plot regret (train)
    for i in range(len(optimizers)):
        plt.plot(np.cumsum(all_losses[i]), label=str(optimizers[i]))
    plt.xlabel('step')
    plt.ylabel('regret')
    plt.grid()
    plt.legend()
    #plt.title('figures/' + name + '_regret_train.png')
    plt.savefig('figures/' + name + '_regret_train.png')
    plt.show()

    # Plot regret
    for i in range(len(optimizers)):
        plt.plot(np.cumsum(all_val_losses[i]), label=str(optimizers[i]))
    plt.xlabel('step')
    plt.ylabel('regret (validation)')
    plt.grid()
    plt.legend()
    #plt.title('figures/' + name + '_regret_valid.png')
    plt.savefig('figures/' + name + '_regret_valid.png')
    plt.show()

    # Plot regret (both)
    for i in range(len(optimizers)):
        plt.plot(np.cumsum(all_losses[i]), label=str(optimizers[i]), c=colors[i])
        plt.plot(np.cumsum(all_val_losses[i]), label='eta:'+str(optimizers[i]), linestyle='--', c=colors[i])
    plt.xlabel('step')
    plt.ylabel('regret (both)')
    plt.grid()
    plt.legend()
    #plt.title('figures/' + name + '_regret_both.png')
    plt.savefig('figures/' + name + '_regret_both.png')
    plt.show()

if __name__ == '__main__':
    main()
