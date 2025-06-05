<h1 align="center">Pytorch Transformer</h1>

<h2>Description:</h2>

<p>
To test my understanding of Transformers, I implemented a decoder-only Transformer (GPT-style) from scratch in PyTorch, based on the "Attention Is All You Need" paper. Due to resource limitations, this model focuses on character-level generation using smaller text datasets, like the Shakespeare's Macbeth script. Some key implemented features are causal masking,  dot-product attention, and multi-head attention. Options for masked and scaled attention are also available, cross attention has been implemented aswell but never tested and so I am unsure if it actually works.

</p>

<h2>Usage:</h2>
<ol>
  <li>Activate a virtual environment.</li>
  <li>Run <code>pip install -r requirements.txt</code> to install the dependencies.</li>
  <li>Run <code>main.py</code> to train a model. (Inference is done in the <code>inference.ipynb</code> Jupyter notebook)</li>
</ol>


<h2>Hyperparameters:</h2>
<p>All hyperparameters are defined in <code>main.py</code>.</p>
<ul>
  <li><code>TXT_FILE_NAME</code> (str): Specifies the text file to use as input.</li>
  <li><code>BATCH_SIZE</code> (int): The number of samples per batch during training.</li>
  <li><code>EPOCHS</code> (int): The number of training epochs.</li>
  <li><code>SHUFFLE</code> (bool): Whether to shuffle the data in the data loaders.</li>
  <li><code>BLOCK_SIZE</code> (int): Defines the fixed length of input sequences during training, and sets the context window size for text generation.</li>
  <li><code>DEVICE</code> (str): The device used to run the code—either <code>"cpu"</code> or <code>"cuda"</code>. Automatically selects <code>"cuda"</code> if available.</li>
  <li><code>WORKERS</code> (int): The number of workers used by the data loader.</li>
  <li><code>VOCAB_SIZE</code> (int): The number of unique characters (tokens) in the dataset.</li>
  <li><code>EMBEDDING_DIM</code> (int): The dimensionality of the character embedding vectors.</li>
  <li><code>NUM_HEADS</code> (int): The number of attention heads in the multi-head attention blocks.</li>
  <li><code>POSITIONAL_FFNN_HIDDEN_NEURONS</code> (int): The number of hidden neurons in the position-wise feed-forward network blocks.</li>
  <li><code>DROPOUT_PROB</code> (float): The dropout probability used within the model.</li>
  <li><code>WORLD_SIZE</code> (int): The number of GPU devices to use (only applicable if <code>cuda</code> is available). This is set to the minimum of the available CUDA GPUs or <code>BATCH_SIZE / 16</code>, with a minimum value of 1.</li>
  <li><code>ALPHA</code> (float): The learning rate.</li>
  <li><code>LOSS_FN</code>: The loss function used during training.</li>
  <li><code>OPTIMIZER</code>: The optimization algorithm used to update model weights.</li>
  <li><code>EXP_NAME</code> (str): The name used to identify the experiment log.</li>
  <li><code>MODEL_NAME</code> (str): The filename used to save the model's parameters.</li>
  <li><code>EVAL_INTERVAL</code> (int): The number of training batches between evaluation logs.</li>
  <li><code>SAVE_INTERVAL</code> (int): The number of training batches between saving logs and model parameters.</li>
</ul>



<h2>Results:</h2>
<p>
  This project was a success! I successfully implemented a decoder Transformer architecture, including multi-head attention with optional masking, scaling, and support for cross-attention (note that while cross-attention is implemented, it was not tested in this project). 
  Below are some of the training results for models trained on three different text sources: Shakespeare's Macbeth, the Bee Movie script, and the Hamilton musical script.
</p>
<p>
  All results, including text generation samples and evaluation metrics, can be found in the <code>inference.ipynb</code> Jupyter notebook.
</p>

<p>
<ul>
  <li>
    </br>The training results below show successful learning across all datasets with increasing accuracy and decreasing loss over time. As illustrated in the graphs, all models effectively learned patterns from their respective texts. Variations in training duration per model are due to the differing sizes of the training datasets.


![clipboard253](https://github.com/user-attachments/assets/d812ba37-1c65-454a-a491-fdf1a89ada07)

  </li>
  <li>
    </br>To show the model's generative capabilities, here's a snippet generated from the Bee Movie script model, using the initial seed text "You like jazz?":


![image](https://github.com/user-attachments/assets/40df88eb-0ae0-4a49-9945-52203c46329d)


  </li>
</ul>
</p>








