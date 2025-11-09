.. _distributed_training:

====================
Distributed Training
====================

Last updated: 2025-11-04


This guide explains how to leverage distributed training to fine-tune
your model on both single-node (multi-GPU) and multi-node
(multi-machine) setups. Our training scripts are built on PyTorch's
distributed capabilities.

Single-Node, Multi-GPU Training
-------------------------------

This is the most common scenario for training on a single machine with
multiple GPUs.

Instructions
------------

Set the ``NPROC_PER_NODE`` environment variable to the number of GPUs
you want to use. Then, execute the training script. The train.sh script
will use this variable to launch the appropriate number of processes.

.. code-block:: bash

   # Set the number of GPUs to use on this machine (e.g., 8)
   export NPROC_PER_NODE=8

   # Run the training script
   # Arguments: <training_script.py> <config_file.yaml>
   export PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH
   sh train.sh tasks/train_llada2_bd.py configs/sft/llada2_mini_bd_sft.yaml

Multi-Node, Multi-GPU Training
-------------------------------

For large-scale training, you can scale across multiple machines. This
requires network communication between the nodes.

Prerequisites
-------------

1. **Network Connectivity:** All nodes must be able to communicate with
   each other over the network. Specifically, all worker nodes must be
   able to reach the ``MASTER_ADDR`` on the specified ``MASTER_PORT``.
2. **Shared Code/Data:** Ensure that the code repository and dataset are
   accessible on all nodes at the same path.

Environment Variables
---------------------

You must configure the following environment variables on **each** node:

- ``NNODES``: The total number of nodes participating in the training.
- ``NODE_RANK``: The unique rank of the current node. This must be 0 for
  the master node and 1, 2, … for the worker nodes.
- ``MASTER_ADDR``: The IP address of the master node (the node with
  NODE_RANK=0).
- ``MASTER_PORT``: A free network port on the master node for
  communication. 29500 is a common default.
- ``NPROC_PER_NODE``: The number of GPUs to use on each node.

Example for a 2-Node Setup
--------------------------

Below is an example of how to launch training on two machines, each with
8 GPUs.

**On the Master Node (IP: 192.168.1.1, Rank: 0):**

Run the following commands in your terminal:

.. code-block:: bash

   # Total number of nodes
   export NNODES=2
   # Rank of this node
   export NODE_RANK=0
   # IP address of this master node
   export MASTER_ADDR="192.168.1.1"
   # Port for communication
   export MASTER_PORT=29500
   # Number of GPUs on this node
   export NPROC_PER_NODE=8

   # Run the training script
   export PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH
   sh train.sh tasks/train_llada2_bd.py configs/sft/llada2_mini_bd_sft.yaml

**On the Worker Node (Rank: 1):**

Run the following commands in your terminal on the second machine:

.. code-block:: bash

   # Total number of nodes (must be the same as on master)
   export NNODES=2
   # Rank of this node (note the change!)
   export NODE_RANK=1
   # IP address of the master node
   export MASTER_ADDR="192.168.1.1"
   # Port on the master node (must be the same)
   export MASTER_PORT=29500
   # Number of GPUs on this node
   export NPROC_PER_NODE=8

   # Run the training script
   export PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH
   sh train.sh tasks/train_llada2_bd.py configs/sft/llada2_mini_bd_sft.yaml

Once the commands are executed on all nodes, the training will begin.
The master node will coordinate the process, and you should see training
logs on all participating machines.
