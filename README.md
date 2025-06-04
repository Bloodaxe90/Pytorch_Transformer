<h1 align="center">Recurrent Networks</h1>

<h2>Description:</h2>

<p>
After completing my PyTorch Transformer project (https://github.com/Bloodaxe90/Pytorch_Transformer), I wanted to revisit and test my understanding of the neural networks traditionally used for sequential data before transformers were discovered. To do this, I implemented an RNN, LSTM, and GRU from scratch using PyTorch (without relying on PyTorch’s built-in implementations) and trained them on the Fashion-MNIST dataset. Another objective of this project was to compare the strengths and weaknesses of each architecture.
</p>
<p>
  <ul>
    My expected observations are:
  <li>
    The vanilla RNN would have a lower accuracy on longer input sequences.
  </li>
  <li>
    The RNN would demonstrate better computational efficiency compared to the LSTM and GRU.
  </li>
  <li>
    The GRU will provide a compromise, giving a better performance than the RNN (especially with longer sequences) while being more computationally efficient than the LSTM.
  </li>
  </ul>
</p>

<h2>Usage:</h2>
<ol>
  <li>Activate a virtual environment.</li>
  <li>Run <code>pip install -r requirements.txt</code> to install the dependencies.</li>
  <li>Run <code>main.py</code> to train and test a model.</li>
</ol>


<h2>Hyperparameters:</h2>
<p>All hyperparameters are defined in <code>main.py</code>.</p>
<ul>
  <li><code>DATASET</code> (str): Specifies the dataset to use — either <code>MNIST</code> or <code>Fashion_MNIST</code>.</li>
  <li><code>BATCH_SIZE</code> (int): The number of samples per batch for training.</li>
  <li><code>EPOCHS</code> (int): The number of training epochs.</li>
  <li><code>SHUFFLE</code> (bool): Whether to shuffle the data in the data loaders.</li>
  <li><code>CHUNK_SIZE</code> (int): Defines the size of each individual segment (chunk) after an image is flattened and divided into a sequence.
    <br>For example, for a 28×28 Fashion-MNIST image (784 total pixels):
    <ul>
      <li>If <code>CHUNK_SIZE = 28</code>, the image is treated as a sequence of 28 chunks, each with 28 pixel values (sequence length = 28, chunk vector size = 28).</li>
      <li>If <code>CHUNK_SIZE = 1</code>, the image is treated as a sequence of 784 chunks, each with 1 pixel value (sequence length = 784, chunk vector size = 1).</li>
    </ul>
  </li>
  <li><code>DEVICE</code> (str): The current device running the code — either <code>"cpu"</code> or <code>"cuda"</code>. Automatically selects <code>"cuda"</code> if available.</li>
  <li><code>SEED</code> (int): Random seed used for reproducibility.</li>
  <li><code>WORKERS</code> (int): The number of workers used by the data loader.</li>
  <li><code>LAYER_TYPE</code> (<code>src.layers</code>): The recurrent layer type to use — RNN, LSTM, or GRU.</li>
  <li><code>NUM_RECURRENT_LAYER</code> (int): The number of stacked recurrent layers.</li>
  <li><code>INPUT_DIM</code> (int): The input dimension — should be equal to <code>CHUNK_SIZE</code>.</li>
  <li><code>HIDDEN_DIM</code> (int): The number of hidden neurons in each recurrent layer.</li>
  <li><code>OUTPUT_DIM</code> (int): The output dimension — already set to the number of classes in the Fashion-MNIST dataset.</li>
  <li><code>DROPOUT_PROB</code> (float): The dropout probability used in the recurrent model.</li>
  <li><code>WORLD_SIZE</code> (int): The number of GPU devices to use (only if <code>cuda</code> is available). It is set to the minimum of the number of available CUDA GPUs or <code>BATCH_SIZE / 16</code>, with a minimum of 1.</li>
  <li><code>ALPHA</code> (float): The learning rate.</li>
  <li><code>OPTIMIZER</code>: The optimization algorithm to use.</li>
  <li><code>LOSS_FN</code>: The loss function to use during training.</li>
  <li><code>EXP_NAME</code> (str): The name used for the experiment log during testing.</li>
  <li><code>MODEL_NAME</code> (str): The filename used to save the model's parameters.</li>
  <li><code>TRAIN_EVAL_INTERVAL</code> (int): The number of batches between each evaluation log during training.</li>
  <li><code>TRAIN_SAVE_INTERVAL</code> (int): The number of batches between each save of logs and model parameters during training.</li>
  <li><code>TEST_EVAL_INTERVAL</code> (int): The number of batches between each evaluation log during testing.</li>
  <li><code>TEST_SAVE_INTERVAL</code> (int): The number of batches between each save of logs during testing.</li>
</ul>


<h2>Results:</h2>
<p>
  This project was very sucessfull and showed the different characteristics of RNN, LSTM, and GRU architectures. The models were identically structured (except for their recurrent layers), trained on Fashion-MNIST with the same data, epochs, batches, and random seed for fair comparison. The experiments were designed to validate initial hypotheses regarding their performance, especially with varying input sequence lengths. The baseline models used (28, 28) input (28 sequences of 28 elements), while the final models used longer sequences of (56, 14) to better show the capabilities of GRU and LSTM against the vanilla RNN.
</p>
<p>
  All results can be found in the <code>inference.ipynb</code> Jupyter notebook
</p>
<p>
<ul>
  <li>
    </br>The training results below show the key performance differences. Specifically, the RNN completed training epochs fastest, confirming its computational efficiency. However, both LSTM and GRU consistently achieved lower final training loss and higher training accuracy. When comparing baseline (28 sequences) to final (56 sequences) results, the LSTM and GRU maintained or improved their performance more substantially than the RNN, indicating their better capacity to learn from longer temporal patterns, with LSTM showing a slightly more pronounced advantage in this training phase.


![clipboard3](https://github.com/user-attachments/assets/8f5cbf89-7a01-42b4-b90b-c8b873f417fe)
![clipboard10](https://github.com/user-attachments/assets/143b7b82-7047-49f5-b740-2612720c56eb)
  </li>
  <li>
    </br>The test results below confirm the training results. On the baseline (28x28) test data the LSTM demonstrated the highest accuracy, followed by GRU, both outperforming the RNN. This performance gap grew on the longer (56x14) test dataset, where the GRU outperformed the LSTM with marginally the best mean accuracy and lowest mean loss. The RNN's performance notably decreased with the increased sequence length, showing its limitations in capturing long term dependencies. (Note that I only test after fully training the models for all epochs)

![clipboard4](https://github.com/user-attachments/assets/8f65ecbc-c827-456e-b02c-331c5fe90966)
![clipboard9](https://github.com/user-attachments/assets/294ca02b-dd1f-4691-907d-d9ffd85dc9aa)

  </li>
</ul>
</p>
</ul>
</p>






