import pathlib
import jraph
import numpy as np
import pandas as pd


def load_data(bulk_data_path, anom_data_path, full_prop_path, length_cut=1000):

  # title pull in big data
  data = np.load(bulk_data_path, allow_pickle=True)

  # pull out target values
  Y = data['arr_1']

  # for datafile header
  columns = ['cnts_aper_b',
  'cnts_aperbkg_b',
  'src_cnts_aper_b',
  'flux_aper_b',
  'hard_hm',
  'hard_hs',
  'hard_ms',
  'var_prob_b',
  'var_prob_h',
  'var_prob_m',
  'var_prob_s']

  # pull out event files: time, energy, pos_x, pos_y
  X = data['arr_0']

  # pull out obsids
  obsids = data['arr_2']

  # finally, load the anomalous objects
  anom_obsids = pd.read_csv(anom_data_path)
  full_prop = pd.read_csv(full_prop_path)

  my_idx = []
  my_ids = []
  for i,id in enumerate(full_prop['obsid'].values):
      if id in full_prop['obsid'].values:
        if full_prop['region_id'].values[i] in anom_obsids['region_id'].values:
          if full_prop['name'].values[i] in anom_obsids['name'].values:
            my_idx.append(i)
            my_ids.append(id)


  my_idx = np.array(my_idx)
  my_ids = np.array(my_ids)

  unique_anomalies, idxs = np.unique(my_ids, return_index=True)
  unique_anomaly_idxs = my_idx[idxs]

  # now get a set of anomalies and regular objects to compare
  X_anom = X[unique_anomaly_idxs]
  Y_anom = Y[unique_anomaly_idxs]

  lengths = np.array([x.shape[0] for x in X_anom])

  X_anom = X_anom[lengths < length_cut]

  # get a set of regular objects
  mask = np.ones(len(X)).astype(bool)
  mask[unique_anomaly_idxs] = False

  # cut X,Y,obsids
  X_reg = X[mask]
  Y_reg = Y[mask]
  reg_obsids = obsids[mask]

  # length cut
  lengths = np.array([x.shape[0] for x in X_reg])
  X_reg = X_reg[lengths < length_cut]
  Y_reg = Y_reg[lengths < length_cut]
  reg_obsids = reg_obsids[lengths < length_cut]

  # rescale the target quantities we care about
  from sklearn.preprocessing import MaxAbsScaler,StandardScaler
  sc = MaxAbsScaler()


  Y_vars = np.nan_to_num(np.array([y[-4:] for y in Y_reg]), nan=0.0) # already from 0 to 1
  Y_hardness = sc.fit_transform(np.nan_to_num(np.array([y[:7] for y in Y_reg]), nan=-1.0)) # rescale 0 to 1

  anom_vars = (np.nan_to_num(np.array([y[-4:] for y in Y_anom]), nan=0.0))
  anom_hardness = sc.transform(np.nan_to_num(np.array([y[:7] for y in Y_anom]), nan=-1.0))


  # now put all Y data into nice pd dataframes

  Y_reg = pd.DataFrame(data=Y_hardness, columns=columns[:7])
  for i,var in enumerate(Y_vars.T):
    Y_reg.insert(7+i, columns[7+i], var)

  # insert the "I=0" anomaly tag
  tag = np.zeros((len(Y_reg),))
  Y_reg.insert(11, 'anomaly_tag', tag)



  # now do the anomalies
  Y_anom = pd.DataFrame(data=anom_hardness, columns=columns[:7])
  for i,var in enumerate(anom_vars.T):
    Y_anom.insert(7+i, columns[7+i], var)

  # insert the "I=1" anomaly tag
  tag = np.ones((len(Y_anom),))
  Y_anom.insert(11, 'anomaly_tag', tag)

  # concatenate the dataframes
  #Y_processed = pd.concat([Y_df, Y_anom])

  return (X_reg, Y_reg), (X_anom, Y_anom)

# data loader script
class DataReader:
  """Data Reader for Open Graph Benchmark datasets."""

  def __init__(self,
            bulk_data_path,
            anom_data_path,
            full_prop_path,
            batch_size=1,
            split_idx=0.75):


    """Initializes the data reader by loading in data."""
    (self.X_reg, self.Y_reg), (self.X_anom, self.Y_anom) = (X_reg, Y_reg), (X_anom, Y_anom)
                                          # load_data(bulk_data_path,
                                          #    anom_data_path,
                                          #    full_prop_path)


    self.Y_anom = self.Y_anom.values[:, :11] # do regression first
    self.Y_reg = self.Y_reg.values[:, :11]


    # split data into train and validation sets
    self._split_idx = int(split_idx*len(self.X_reg))

    self.X_train = self.X_reg[:self._split_idx]
    self.X_val = self.X_reg[self._split_idx:]; del self.X_reg

    self.Y_train = self.Y_reg[:self._split_idx]
    self.Y_val = self.Y_reg[self._split_idx:]; del self.Y_reg


    # compute the number of nodes we will have per event file
    self.numDqs = lambda l: l*(l-1) / 2. # function giving us number of nodes # n(n-1)/2

    # set training mode (defaults to "train")
    self.switch_training_mode()


    self._repeat = False
    self._batch_size = batch_size
    self._generator = self._make_generator()





  def switch_training_mode(self, mode='train', verbose=False):
    if mode == 'train':
      if verbose:
        print('setting reader to training mode')

      self.Y_dat = self.Y_train
      self.X_dat = self.X_train

    else:
      if verbose:
          print('setting reader to validation mode')

      self.Y_dat = self.Y_val
      self.X_dat = self.X_val

    # re-compute node numbers
    self.get_node_nums()



  def get_node_nums(self):
    # compute the number of nodes we will have per event file
    self._event_lengths = [np.array(len(x)).astype(np.int) for x in self.X_dat]
    self._n_node = [self.numDqs(n).astype(int) for n in self._event_lengths]

    # If n_node = [1,2,3], we create accumulated n_node [0,1,3,6] for indexing.
    self._accumulated_n_nodes = np.concatenate((np.array([0]),
                                                np.cumsum(self._n_node)))

  @property
  def total_num_graphs(self):
    return len(self._n_node)

  def repeat(self):
    self._repeat = True

  def __iter__(self):
    return self

  def __next__(self):
    graphs = []
    labels = []
    for _ in range(self._batch_size):
      graph, label = next(self._generator)
      graphs.append(graph)
      labels.append(label)
    return jraph.batch(graphs), np.concatenate([labels], axis=0)

  def get_graph_by_idx(self, idx):
    """Gets a graph by an integer index."""
    # Gather the graph information
    label = self.Y_dat[idx] #self._labels[idx]

    # get the numbe of nodes
    n_node = self._n_node[idx]

    # get the (dt,de,dx,dy) node data
    node_dat = self._make_event_set(self.X_dat[idx])

    # node_slice = slice(
    #     self._accumulated_n_nodes[idx], self._accumulated_n_nodes[idx+1])
    # edge_slice = slice(
    #     self._accumulated_n_edges[idx], self._accumulated_n_edges[idx+1])
    # nodes = self._nodes[node_slice]

    return jraph.GraphsTuple(
        n_node=jnp.asarray([n_node]), n_edge=jnp.asarray([0]),
        nodes=node_dat, edges=jnp.asarray([0]),
        globals={},
        senders=jnp.asarray([0]), receivers=jnp.asarray([0])), label


  def _make_event_set(self, dat):

    skip = 1

    # grid the data
    ts = dat.T[0][::skip]
    eps = dat.T[1][::skip]
    xs = dat.T[2][::skip]
    ys = dat.T[3][::skip]

    n = len(ts)

    dt_grid = []
    de_grid = []
    dx_grid = []
    dy_grid = []

    # loop over all pairs
    for i in range(n):
        # for t
        tdiff = (ts - ts[i])[i+1:]
        dt_grid.append(tdiff)

        # for eps
        ediff = (eps[i] - eps)[i+1:]
        de_grid.append(ediff)

        # for x
        xdiff = (xs[i] - xs)[i+1:]
        dx_grid.append(xdiff)

        # for y
        ydiff = (ys[i] - ys)[i+1:]
        dy_grid.append(ydiff)

    dts = np.concatenate(dt_grid)
    des = np.concatenate(de_grid)
    dxs = np.concatenate(dx_grid)
    dys = np.concatenate(dy_grid)

    return np.stack((dts,des,dxs,dys)).T


  def _make_generator(self):
    """Makes a single example generator of the loaded OGB data."""
    idx = 0
    while True:
      # If not repeating, exit when we've cycled through all the graphs.
      # Only return graphs within the split.
      if not self._repeat:
        if idx == self.total_num_graphs:
          return
      else:
        # This will reset the index to 0 if we are at the end of the dataset.
        idx = idx % self.total_num_graphs
      #if idx not in self._split_idx:
      idx += 1
        #continue
      graph, label = self.get_graph_by_idx(idx)
      idx += 1
      yield graph, label
