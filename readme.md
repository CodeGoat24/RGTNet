# RGTNet

## Dataset

### ABIDE

Please follow the [instruction](util/abide/readme.md) to download and process this dataset.

## Usage

### ABIDE 

```bash
python main.py --config_filename setting/abide_RGTNet.yaml
```

## Hyper parameters

All hyper parameters can be tuned in setting files.

```yaml
model:
  
  type: GraphTransformer

  # gru or cnn or attention or gau
  extractor_type: attention
  embedding_size: 8
  gcn_layer: 4
  window_size: 4

  cnn_pool_size: 16

  # product or linear
  graph_generation: product

  num_gru_layers: 4

  dropout: 0.5



train:
  # normal or bilevel 
  method: normal
  lr: 1.0e-4
  weight_decay: 1.0e-4
  epochs: 500
  pool_ratio: 0.7
  optimizer: adam
  stepsize: 200

  group_loss: true
  sparsity_loss: true
#  sparsity_loss_weight: 1.0e-4
  sparsity_loss_weight: 0.5e-4
  log_folder: /result
  
  # uniform or pearson
  pure_gnn_graph: pearson
```

# Model Zoo
We provide models for RGTNet_AAL and RGTNet_CC200.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>atlas</th>
      <th>acc.%</th>
      <th>sen.%</th>
      <th>spe.%</th>
      <th>auc.%</th>
      <th>url of model</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAL</td>
      <td>73.4</td>
      <td>70.8</td>
      <td>71.9</td>
      <td>72.3</td>
      <td><a href="https://pan.baidu.com/s/1k2xtr9OCdEiSFWrYJVXcww">baidu disk</a>&nbsp;(code: sfrp)</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>CC200</td>
      <td>74.4</td>
      <td>75.2</td>
      <td>73.4</td>
      <td>76.0</td>
      <td><a href="https://pan.baidu.com/s/1QdI4EBKdg1IGS_XTMD9z4Q">baidu disk</a>&nbsp;(code: apmu)</td>
      <td></td>
    </tr>

  </tbody>
</table>

