---
title: Aprendizaje federado jerárquico para la predicción energética
url: None
labels: [hierarchical federated learning, energy prediction]
dataset: [PLEIAData]
---

> [!IMPORTANT]
> This is the template for your `README.md`. Please fill-in the information in all areas with a :warning: symbol.
> Please refer to the [Flower Baselines contribution](https://flower.ai/docs/baselines/how-to-contribute-baselines.html) and [Flower Baselines usage](https://flower.ai/docs/baselines/how-to-use-baselines.html) guides for more details.
> Please complete the metadata section at the very top of this README. This generates a table at the top of the file that will facilitate indexing baselines.
> Please remove this [!IMPORTANT] block once you are done with your `README.md` as well as all the `:warning:` symbols and the comments next to them.

> [!IMPORTANT]
> To help having all baselines similarly formatted and structured, we have included two scripts in `baselines/dev` that when run will format your code and run some tests checking if it's formatted.
> These checks use standard packages such as `isort`, `black`, `pylint` and others. You as a baseline creator will need to install additional packages. These are already specified in the `pyproject.toml` of
> your baseline. Follow these steps:

```bash
# Create a python env
pyenv virtualenv 3.12.12 baseline

# Activate it
pyenv activate baseline

# Install project including developer packages
# Note the `-e` this means you install it in editable mode 
# so even if you change the code you don't need to do `pip install`
# again. However, if you add a new dependency to `pyproject.toml` you
# will need to re-run the command below
pip install -e ".[dev]"

# Even without modifying or adding new code, you can run your baseline
# with the placeholder code generated when you did `flwr new`. If you
# want to test this to familiarise yourself with how flower apps are
# executed, execute this from the directory where you `pyproject.toml` is:
flwr run .

# At anypoint during the process of creating your baseline you can 
# run the formatting script. For this do:
cd .. # so you are in the `flower/baselines` directory

# Run the formatting script (it will auto-correct issues if possible)
./dev/format-baseline.sh baseline

# Then, if the above is all good, run the tests.
./dev/test-baseline.sh baseline
```

> [!IMPORTANT]
> When you open a PR to get the baseline merged into the main Flower repository, the `./dev/test-baseline.sh` script will run. Only if test pass, the baseline can be merged. 
> Some issues highlighted by the tests script are easier than others to fix. Do not hesitate in reaching out for help to us (e.g. as a comment in your PR) if you are stuck with these.
> Before opening your PR, please remove the code snippet above as well all the [!IMPORTANT] message blocks. Yes, including this one.

# PleiadesHVAC

**Paper:** Present in Paper directory if you speak spanish. Alternatively you can read the Extended Abstract section of this file.

**Authors:** Alfredo Marquina Meseguer

**Abstract:** This abstract is temporary. This bachelor's thesis tries to implement hierarchical federated learning (HFL) for the prediction of power consumption in Flower. Because there is no proper way to implement the aggregator element needed for implementing HFL, the proposed architecture simulates an aggregator with a client and server each one present in a different federation, simulating a three level HFL architecture.

## About this baseline

**What’s implemented:** :warning: *_Concisely describe what experiment(s) (e.g. Figure 1, Table 2, etc.) in the publication can be replicated by running the code. Please only use a few sentences. ”_*

**Datasets:**  [Pleidata](https://zenodo.org/records/7620136)

**Hardware Setup:** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Indicate how long it took to run the experiments. Someone out there might not have access to the same resources you have so, could you list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

**Contributors:** Juan de Dios Hernández Kakauridze, 


## Extended Abstract

In the contemporary era of smart cities and sustainable development, optimization of energy consumption within buildings has emerged as a critical socio-economic and environmental priority. Consequently, the capacity to accurately predict energy demand patterns constitutes a foundational pillar in reducing environmental impacts and operational costs.

With the proliferation of Internet of Things (IoT) infrastructures, modern smart buildings are continuously monitored by heterogeneous wireless sensor networks. These devices capture continuous data streams obtaining all kinds of information. To transform this raw data into actionable intelligence, Deep Learning (DL) methodologies have been widely adopted due to their capability to model non-linear, complex spatio-temporal dependencies inherent to multi-variable time-series forecasting.

However, the traditional deployment of DL frameworks relies heavily on centralized cloud-computing paradigms where all raw data collected by localized edge sensors must be continuously transmitted over the network to a monolithic cloud repository where the model is trained. This operational blueprint introduces severe structural bottlenecks: data privacy, due to the highly sensitive nature of some data; cybersecurity risks, because an attack to the centralized server may expose all data as well as be susceptible to man-in-the-middle attacks; and bandwidth constraint, because continuous flow of data may introduce excessive overhead in the network.

To definitively alleviate these limitations, Federated Learning (FL) has emerged as a disruptive decentralized machine learning paradigm. In standard FL, data remains strictly localized on the generating edge devices or clients. Instead of transmitting raw datasets, clients train a local instance of the model using their proprietary data and exclusively upload the resulting model parameters (weights and biases) to a centralized server. The server aggregates these local updates utilizing a so-called aggregation algorithm (such as Federated Averaging or FedAvg) to update a shared global model, which is subsequently redistributed back to the clients. While FL natively guarantees data privacy and drastically minimizes network bandwidth consumption, standard two-tier FL topologies (client-to-server) may face scalability and compatibility issues when applied to already existing infrastructures such as the three-layer topologies seen in many edge computing environments used nowadays, especially those used in medical settings.

To bridge this gap, Hierarchical Federated Learning (HFL) has been proposed in literature as a natural architectural evolution. HFL introduces intermediate aggregation nodes, typically deployed at the edge layer, establishing a multi-tier hierarchical infrastructure. HFL achieves this goal with the addition of the aggregator node, an intermediate stackable element between servers and clients.
Following a current line of investigation by the University of Murcia on the application of FL to the such predictions of energy consumption while using the dataset PLEIAData which compiles a wide range of climatization and energy consumption data compiled through the year of 2021 in the smart building PLEIADES, also from the University of Murcia. This Bachelor’s Thesis tries to give continuity to the work started by Hernández with implementation and testing of a three-tier HFL federation. The project has been developed in Python using the most widely used FL framework Flower and Tensorflow for the DL models.

This intermediate tier acts as a localized buffer and regularizer. Instead of immediately averaging highly divergent local updates at the global scale, edge aggregators consolidate the updates of logical or physical clusters (e.g., specific floors or independent buildings blocks). This allows the network to capture regional contextual commonalities before abstracting the model weights to the global cloud server. As a side effect this architecture reduces the communication frequency between local edge facilities and distant cloud architectures which tend to be more limited and costly.

Concurrently, selecting the optimal underlying neural network architecture for edge-oriented time-series forecasting remains a critical design challenge. While traditional Long Short-Term Memory (LSTM) networks have long been the gold standard for sequence modeling due to their gating mechanisms designed to mitigate the vanishing gradient problem. LSTMs utilize three distinct gating structures (input, forget, and output gates) fed by two distinct channels one for traditional input and another one for memory information.

However, the search for greater efficiency has landed in the creation of the Gated Recurrent Units (GRU) architectures.  A GRU couples the forget and input gates into a single update gate and merges the cell state and hidden state, operating with only two gates (update and reset gates) and fusing both channels into one. Mathematically, the reduction in internal gates translates into a significantly lower parameter count.
This architectural simplification directly implies reduced RAM utilization, lower energy consumption during local processing cycles, and faster execution times. Despite having fewer parameters, the literature demonstrates that GRUs maintain a competitive performance profile closely matching LSTMs in specific short-to-medium-term regression tasks. 

Furthermore, highly sophisticated architectures like Convolutional LSTMs (ConvLSTM2D), which capture coupled spatial and temporal dynamics, and Transformer networks based on Self-Attention mechanisms offer state-of-the-art accuracy in centralized environments, have been tested to evaluate their performance in HFL environments following \citeauthor{hernandez_kakaurize_uso_2025}’s work, who implemented these models.

The experimental validation of this research relies on PLEIAData, a comprehensive multi-variable dataset gathered throughout the year 2021 from the PLEIADES experimental smart building located at the University of Murcia. This dataset builds upon foundational frameworks from prior work, maintaining consistency in data definitions while adapting the workflow to optimized pipelines. PLEIAData records high-frequency readings across diverse architectural zones, fundamentally divided into three autonomous structural building segments: Block A, Block B, and Block C.

To guarantee stable weight optimization during local model updates, a rigorous data preprocessing pipeline was developed by \citeauthor{hernandez_kakaurize_uso_2025} in the previous work. After selecting carefully the included dataset variables, a global Min-Max Normalization was applied. Following normalization, the data is transformed into three-dimensional arrays suitable for recurrent operations. In previous iterations of this research, massive history windows (such as 168 hours) were enforced, to accommodate resource-constrained IoT requirements, this study reduced the temporal window size down to 12 hours. This reduction preserves immediate context while reducing the historical parameter steps, striking an optimal balance between context preservation and memory footprint. This data was already horizontally segmented across clients representing localized blocks in the original PLEIAData dataset, each segment being used as a dataset for local clusters.

A primary milestone of this thesis consists of the complete migration and re-implementation of the legacy simulation code into the modern Flower framework (version 1.30). However, Flower’s native simulation engine operates strictly on a traditional two-tier hub-and-spoke layout. It does not inherently support intermediate aggregation rings. To successfully bypass this operational restriction without introducing networking conflicts, a decoupled execution flow was engineered.

The three-tier Hierarchical Federated Learning (HFL) layout is instantiated by configuring intermediate ClientApp and ServerApp routines to act as proxy edge aggregators for specific building blocks. In this workflow, local clients train directly within their local cluster boundaries (their own block). Once local updates settle, the intermediate proxy aggregators consolidate their cluster-specific weights using local instances of the Federated Averaging (FedAvg) algorithm. These block models are then sent to the root cloud server for final global averaging. This design effectively overrides Flower's default constraints, establishing an isolated, hierarchical paradigm tailored to the physical building subdivisions.

To accurately benchmark the performance gains of structural model optimization, the newly proposed models are directly compared against three established baselines inherited from previous research lines:

- Long Short-Term Memory (LSTM): A classic sequence model utilizing input, forget, and output gates to track temporal consumption states across sequences. It serves as the primary predictive standard.
- Convolutional LSTM (ConvLSTM2D): A hybrid model that stacks spatial convolutional operations inside recurrent cell transitions, designed to isolate inter-variable dependencies along the sequence path.
- Transformer Networks: An architecture built entirely around Multi-Head Self-Attention layers.

The integration of Gated Recurrent Unit (GRU) alternatives directly addresses the necessity of maximizing edge execution efficiency within resource-constrained IoT settings. As previously explained, GRU cells reduce their internal complexity compared to their LSTM counterparts. Two specific GRU variants were designed, compiled via TensorFlow/Keras, and integrated into the HFL workflow:

- Standard GRU Architecture: Mirroring inherited LSTM architecture, this architecture substitutes just both LSTM with GRU layers. The final product consists of just two GRU layers and two dense layers. Regularization was applied using LayerNormalization and Dropout to prevent overfitting.
- Simplified GRU Architecture: This configuration consists of only two GRU layers as well as the input and output. This pruning optimizes the model as much as possible.

The comprehensive evaluation of the decentralized framework was conducted through controlled simulations consisting of 10 synchronized global training rounds, with local edge training configured at 2 internal epochs per round. For each model presented, two distinct executions were made: the first one for the evaluation of the model in a HFL paradigm and the second one for evaluation of the same model in a FL paradigm with the purpose of comparison. For the HFL experiment three distinct local clusters were created to mirror real world organization of the PLEIADES building into blocks, meanwhile the FL experiment created a combined dataset of all three blocks. Performance was benchmarked using statistical regression metrics, primarily the Coefficient of Determination (R2 Score), Mean Absolute Error (MAE), and Mean Squared Error (MSE).

In the first experimental phase, each one of the local clusters presented their own performance along the aggregated model’s performance. Local cluster performance  seemed to be ordered by the quality of their local dataset which consistently were: A as first, B as a close second and C as the last one. All models showed a solid learning curve barring ConvLSTM2D’s model which showed signs of divergence even though its metrics were similar (MSE = 0.028) to those obtained in previous works (MSE=0.03) where it showed a healthy learning curve.

Second experimental phase, compared all the different aggregated models during their training. The outcomes obtained by this experiment resulted in a clear win for the simplified GRU architecture. This architecture obtained the best results in most metrics measured (R2 = 0.75, MSE = 0.004 and MAE = 0.038). On the same note the rest of RNN architectures (LSTM and GRU), as well as the attention model (Transformer), showed intertwined learning curves obtaining a similar performance. Finally ConvLSTM2D, continued with its already described behavior.

Finally, the third experimental phase compared the data obtained with these HFL models with their FL counterparts. This phase showed that the structural implementation of the three-tier HFL paradigm outperformed the FL configuration in this scenario. Whereas expected results were slight performance degradation inherent to aggregation functions like the one used, FedAvg. HFL models started with a higher error rate but quickly catched up with FL counterparts up to the point of overcoming them. The only exception to this phenomenon is again ConvLSTM2D, which had a close but higher degree of divergence in the HFL paradigm. This behavior can be explained by the reduction of its input dimensionality as similar results were obtained in both paradigms.

It is hypothesized that in standard FL, the worst performance comes from the variations between blocks that introduce a level of high statistical heterogeneity (non-IID data) across clients inducing model shaking or gradient cancellations. Under the proposed HFL layout, the intermediate proxy edge aggregators act as a structural localized regularizer. Allowing the local cluster to approximate a local model before aggregating to the global one.

Finally, through the measuring of the Max Error of the models it was found that the models obtained catastrophic errors even while maintaining their high performance in other metrics. It is hypothesized that this behavior is caused by the existence of a few non-IID evaluation examples that could be caused by the processing of the dataset or due to the inherent nature of climate data. Either way further research would be required to reach a satisfactory conclusion.

In conclusion, this Bachelor’s Thesis demonstrates the viability and technical advantages of implementing a three-tier HFL architecture for short-term energy prediction within the smart building of PLEIADES. The evaluation validates that lightweight recurrent configurations, specifically the proposed simplified GRU, offer an optimal design pattern for resource constricted environments. 

Future research paths could focus on approaching real implementation though more realistic simulations (for example, with Dockers deployed in a local network) or the implementation of more features either aimed at the improvement of the IoT environment (like online training) or the HFL architecture (through implementation of heterogeneous hierarchical federated learning where a given server or aggregator may have both aggregator and edge node clients).

## Experimental Setup

**Task:** :warning: *_what’s the primary task that is being federated? (e.g. image classification, next-word prediction). If you have experiments for several, please list them_*

**Model:** :warning: *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._*

**Dataset:** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

**Training Hyperparameters:** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

:warning: _Specify the steps to create and activate your environment and install the baseline project. Most baselines are expected to require minimal steps as shown below. These instructions should be comprehensive enough so anyone can run them (if non standard, describe them step-by-step)._

:warning: _The dependencies for your baseline are listed in the `pyproject.toml`, extend it with additional packages needed for your baseline._

:warning: _Baselines should use Python 3.12, [pyenv](https://github.com/pyenv/pyenv), and the [virtualenv](https://github.com/pyenv/pyenv-virtualenv) plugging. 

```bash
# Create the virtual environment
pyenv virtualenv 3.12.12 <name-of-your-baseline-env>

# Activate it
pyenv activate <name-of-your-baseline-env>

# Install the baseline
pip install -e .
```

:warning: _If your baseline requires running some script before starting an experiment, please indicate so here_.

## Running the Experiments

:warning: _Make sure you have adjusted the `client-resources` in the federation in `pyproject.toml` so your simulation makes the best use of the system resources available._

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

:warning: _You might want to add more hyperparameters and settings for your baseline. You can do so by extending `[tool.flwr.app.config]` in `pyproject.toml`. In addition, you can create a new `.toml` file that can be passed with the `--run-config` command (see below an example) to override several config values **already present** in `pyproject.toml`._
```bash
# it is likely that for one experiment you need to override some arguments.
flwr run . --run-config learning-rate=0.1,coefficient=0.123

# or you might want to load different `.toml` configs all together:
flwr run . --run-config <my-big-experiment-config>.toml
```

:warning: _It is preferable to show a single command (or multiple commands if they belong to the same experiment) and then a table/plot with the expected results, instead of showing all the commands first and then all the results/plots._
:warning: _If you present plots or other figures, please include either a Jupyter notebook showing how to create them or include a utility function that can be called after the experiments finish running._
:warning: If you include plots or figures, save them in `.png` format and place them in a new directory named `_static` at the same level as your `README.md`.
