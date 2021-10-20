from tqdm import tqdm
import jraph
import jax.numpy as jnp
import jax

import haiku as hk






def train(batch_size, num_training_steps):
    reader = DataReader(bulk_data_path,
                      anom_data_path,
                      full_prop_path, batch_size=batch_size)

    reader.switch_training_mode('train', verbose=True)
    # repeat dataset forever
    reader.repeat()

    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.without_apply_rng(hk.transform(net_fn))
    # Get a candidate graph and label to initialize the network.
    graph, _ = reader.get_graph_by_idx(0)

    print(graph.n_node)

    # pad graph
    graph = pad_graph_to_nearest_power_of_two(graph)

    # initialize network

    params = net.init(jax.random.PRNGKey(34), graph)
    # Initialize the optimizer.
    opt_init, opt_update = optax.adam(1e-4)
    opt_state = opt_init(params)

    compute_loss_fn = functools.partial(compute_loss, net=net)
    # We jit the computation of our loss, since this is the main computation.
    # Using jax.jit means that we will use a single accelerator. If you want
    # to use more than 1 accelerator, use jax.pmap. More information can be
    # found in the jax documentation.
    compute_loss_fn = jax.jit(jax.value_and_grad(
        compute_loss_fn, has_aux=True))


    # initialize losses
    train_loss = []
    val_loss = []


    pbar = tqdm(range(num_training_steps), position=0, leave=True)

    for idx in pbar:
        graph, label = next(reader)
        # Jax will re-jit your graphnet every time a new graph shape is encountered.
        # In the limit, this means a new compilation every training step, which
        # will result in *extremely* slow training. To prevent this, pad each
        # batch of graphs to the nearest power of two. Since jax maintains a cache
        # of compiled programs, the compilation cost is amortized.
        graph = pad_graph_to_nearest_power_of_two(graph)

        # Since padding is implemented with pad_with_graphs, an extra graph has
        # been added to the batch, which means there should be an extra label.
        label = jnp.concatenate([label, jnp.zeros((1,11))], axis=0)

        #print(label.shape)

        (loss, acc), grad = compute_loss_fn(params, graph, label)
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        train_loss.append(loss)

        if idx % 10 == 0:
          # every 10th iteration do a validation loss calculation (don't update optimizer)
          reader.switch_training_mode('val')
          graph, label = next(reader)
          graph = pad_graph_to_nearest_power_of_two(graph)
          label = jnp.concatenate([label, jnp.zeros((1,11))], axis=0)

          (loss, acc), grad = compute_loss_fn(params, graph, label)

          val_loss.append(loss)
          # switch back to training mode
          reader.switch_training_mode('train')

        # update progress bar
        pbar.set_description("eval epoch: {}, loss: {:.4f}, val loss: {:.4f}".format(
            idx + 1, train_loss[-1], val_loss[-1]
        ))


    return (train_loss,val_loss), params

