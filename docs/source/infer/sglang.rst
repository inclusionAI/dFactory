SGLang dLLM Inference Guide
===========================

Overview
--------

The dLLM (diffusion language model) paradigm is rapidly evolving, and inference ecosystems are still maturing. This guide provides a practical approach to launching dLLM inference services using SGLang.

Installation
------------

Install the specific SGLang version with dLLM support:

.. code-block:: bash

   pip install git+https://github.com/sgl-project/sglang.git@refs/pull/12588/head

Server Launch
-------------

Launch the SGLang inference server with dLLM-specific parameters:

.. code-block:: bash

   python3 -m sglang.launch_server \
       --model-path /path/to/LLaDA2.0-flash-preview/ \
       --host 127.0.0.1 \
       --port 8188 \
       --trust-remote-code \
       --disable-cuda-graph \
       --disable-radix-cache \
       --mem-fraction-static 0.9 \
       --attention-backend flashinfer \
       --diffusion-algorithm "LowConfidence" \
       --diffusion-block-size 32 \
       --tp-size 4 \
       --max-running-requests 1

Key Parameters
~~~~~~~~~~~~~~

- ``--diffusion-algorithm``: Set to "LowConfidence" for dLLM inference
- ``--diffusion-block-size``: Block size for diffusion generation (default: 32)
- ``--attention-backend``: Use "flashinfer" for optimal performance
- ``--tp-size``: Tensor parallelism size (adjust based on GPU count)
- ``--max-running-requests``: Limit concurrent requests for stability

API Usage
---------

Send requests to the inference endpoint:

.. code-block:: bash

   curl -X POST "http://127.0.0.1:8188/generate" \
       -H "Content-Type: application/json" \
       -d '{
           "text": "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Why did Camus say that Sisyphus was happy?<|role_end|><role>ASSISTANT</role>",
           "stream": true,
           "sampling_params": {
               "temperature": 0,
               "max_new_tokens": 1024
           }
       }'

Request Format
~~~~~~~~~~~~~~

- ``text``: Input text with role-based formatting
- ``stream``: Enable streaming responses
- ``sampling_params``: Generation parameters

  - ``temperature``: Sampling temperature (0 for deterministic)
  - ``max_new_tokens``: Maximum tokens to generate

Additional Resources
--------------------

For detailed implementation discussion and RFC, see:
https://github.com/sgl-project/sglang/issues/12766