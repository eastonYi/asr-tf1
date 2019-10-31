# Unsupervised phone classification

## Environments
- Tensorflow == 1.14
- Python == 3.7


## model code
- `__call__`
the forward compution of model and generate logits, which is then used in `build_single_graph` to compute loss and gradients.

- `build_single_graph`
process input and call the self to build forward graph, then compute loss and gradients.
```python
...
with tf.name_scope("gradients"):
    assert loss.get_shape().ndims == 1
    loss = tf.reduce_mean(loss)
    gradients = self.optimizer.compute_gradients(loss, var_list=self.trainable_variables())
```
    - the loss before computing the gradients should be batch_shape
    - no need to add scope here since the scope will be added in `build_graph`



## Exps
Performance of one layer fully-connected classifier:

|Corpus| Supervised | EODM | GAN | GAN + 250 paired | EODM + 250 paired |
|:-----:|-------------|---|:-----:| :-----: | :-----: |
| TIMIT | 26.5 | 40 | 48 | 35.5 | 33.4 |
| AIShell-2 |  30.52 |  - | -  | 42.5  |   |
| LibriSpeech | 28.0  | -  | -  | 41.2  |   |
