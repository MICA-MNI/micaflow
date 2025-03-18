Examples
========

Example Workflows
---------------

Here are some example workflows to help you get started with MicaFlow.

Structural MRI Processing
------------------------

Running basic T1w processing:

.. code-block:: bash

   micaflow pipeline --subject sub-001 --session ses-01 \
     --data-directory /data --t1w-file sub-001_ses-01_T1w.nii.gz \
     --out-dir /output --cores 4

Diffusion MRI Processing
----------------------

Complete DWI processing pipeline:

.. code-block:: bash

   micaflow pipeline --subject sub-001 --session ses-01 \
     --data-directory /data --t1w-file sub-001_ses-01_T1w.nii.gz \
     --run-dwi --dwi-file sub-001_ses-01_dwi.nii.gz \
     --bval-file sub-001_ses-01_dwi.bval --bvec-file sub-001_ses-01_dwi.bvec \
     --inverse-dwi-file sub-001_ses-01_acq-PA_dwi.nii.gz \
     --out-dir /output --cores 4

Registration Example
------------------

This example shows how to use SynthSeg for contrast-agnostic registration:

.. literalinclude:: ../../examples/synthseg_registration.py
   :language: python
   :linenos: