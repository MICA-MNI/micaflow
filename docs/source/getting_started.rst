Getting Started
==============

Installation
-----------

You can install MicaFlow directly via pip:

.. code-block:: bash

   pip install micaflow
   # Verify installation
   micaflow

Or, install from source:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/MICA-MNI/micaflow.git
   cd micaflow

   # Install the package
   pip install -e .

Basic Usage
----------

MicaFlow can be used as a complete pipeline or as individual modules:

.. code-block:: bash

   # Run the full pipeline for a single subject
   micaflow pipeline --subject sub-001 --session ses-01 \
     --data-directory /path/to/data --t1w-file sub-001_ses-01_T1w.nii.gz \
     --output /output --cores 4

Batch Processing (BIDS)
-----------------------

To process an entire BIDS dataset automatically, you can use the batch command. This will scan the BIDS directory for valid subjects/sessions, identify required files based on suffixes, run the pipeline sequentially, and generate a ``micaflow_runs_summary.json`` file:

.. code-block:: bash

   micaflow bids --bids-dir /path/to/bids_root --output-dir /path/to/derivatives \
     --cores 4 --gpu

You can restrict processing to specific subsets using ``--participant-label`` (e.g., ``001 002``) and ``--session-label``.

Dependencies
-----------

MicaFlow requires:

* Python 3.9, 3.10, or 3.11