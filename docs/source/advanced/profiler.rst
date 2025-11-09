.. _profiler:

========
Profiler
========

Last updated: 2025-11-04


VeOmni offers a profiler function for users to trace training:

.. code-block:: python

   from veomni.utils import helper

   # Before train loop, create your profiler
   if args.train.global_rank == 0:
       if args.train.enable_profiling:
           profiler = helper.create_profiler(
               start_step=args.train.profile_start_step,
               end_step=args.train.profile_end_step,
               trace_dir=args.train.profile_trace_dir,
               record_shapes=args.train.profile_record_shapes,
               profile_memory=args.train.profile_profile_memory,
               with_stack=args.train.profile_with_stack,
           )
           profiler.start()

   for epoch in range(args.train.num_train_epochs):
       data_iterator = iter(train_dataloader)
       for _ in range(args.train.train_steps):
           ...  # Your training code
           profiler.step()
           if global_step == args.train.profile_end_step:
               profiler.stop()
               # upload file to merlin
               helper.upload_trace(args.train.wandb_project, args.train.wandb_name, args.train.profile_trace_dir)
