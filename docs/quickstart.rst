Quick Start
============

**torch-ttt** enables the seamless integration of *Test-Time Training* (TTT) methods into your models, offering a flexible and user-friendly approach to enhance model generalization and improve performance on *out-of-distribution* data. 

.. figure:: _static/images/teaser.svg
   :alt: Description of the SVG image
   :align: center
   :width: 100%

   **Figure 1.** *Test-Time Training* (TTT) improves model predictions during inference by optimizing them on the input. **torch-ttt** enables TTT through lightweight *Engine* wrappers that implement specific methods.

The core idea behind this library is that different TTT methods, at a high level, primarily differ in how they compute their self-supervised losses. Conceptually, any TTT method can be abstracted as a black box that takes a model and input, returns a self-supervised loss, and enables further optimization to enhance the model's performance. We call such an abstraction an *"Engine"* and most of the library's functionality is centered around them.

.. figure:: _static/images/teaser_ssl_schema.svg
   :alt: Description of the SVG image
   :align: center
   :width: 100%

   **Figure 2.** Any TTT method can be abstracted as a black box that computes a self-supervised loss and a model prediction. The self-supervised loss is then used for further optimization to improve model performance.

Training with Engines
-----------------------

With an easy-to-use API (centered around *Engines*), you can effortlessly implement, experiment with, and incorporate *Test-Time Training* methods into your training and inference pipelines. An Engine encapsulates the logic of a specific TTT method, seamlessly managing both its training and inference processes.

Getting started is straightforward: during training, simply wrap your model with the chosen Engine class and use it for inference. The Engine will return the model's outputs along with the TTT loss, which should be added to your main loss function.

.. code-block:: diff

   +from torch_ttt.engine.ttt_engine import TTTEngine

   model = MyModel()
   +engine = TTTEngine(model, features_layer_name='layer2') 

   ...Setting up data, optimizers, etc...

   # Training loop
   - model.train()
   + engine.train() # Don't forget it, it's important!
   for batch in train_loader:
         inputs, targets = batch
         optimizer.zero_grad()

   -      outputs = model(inputs)
   +      outputs, loss_ttt = engine(inputs)

   -     loss = loss_fn(outputs, targets)
   +     loss = loss_fn(outputs, targets) + alpha * loss_ttt
   
        loss.backward()
        optimizer.step()

.. important::

    Some TTT methods require an additional step to be performed between training and testing, such as feature statistics calculation for :obj:`TTTPPEngine <torch_ttt.engine.ttt_pp_engine.TTTPPEngine>` and :obj:`ActMADEngine <torch_ttt.engine.actmad_engine.ActMADEngine>`.


During inference, use the Engine with the `run_ttt` function to adapt the model. This function applies TTT-based gradient optimization to adjust the model to the provided inputs, thereby enhancing its performance.

.. code-block:: diff

   # Testing loop
   - model.eval()
   + engine.eval() # Also, don't forget it!
   for batch in test_loader:
         inputs, targets = batch
         optimizer.zero_grad()

   -      outputs = model(inputs)
   +      outputs = engine(inputs)
         metric = compute_metric(outputs, targets)
