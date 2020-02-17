Customization
=================

Cyclum allows customization of the model. You may derive the base class and build your own model.

The default model is actually build by running:

.. code:: python

   encoder = nonlinear_encoder([0], [30, 20]) + linear_encoder([1])
   decoder = circular_decoder([0]) + linear_decoder([1])
   build(encoder, decoder)

We have documented the implementation thoroughly, so you may refer to the documentation.