Getting Started
==============

Installation
-----------

Clone the repository:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/MICA-LAB/micaflow.git
   cd micaflow

   # Install the package
   pip install -e .

Basic Usage
----------

MicaFlow can be used as a complete pipeline or as individual modules:

.. code-block:: bash

   # Run the full pipeline
   micaflow pipeline --subject sub-001 --session ses-01 \
     --data-directory /path/to/data --t1w-file sub-001_ses-01_T1w.nii.gz \
     --out-dir /output --cores 4

Dependencies
-----------

MicaFlow requires:

* Python 3.9, 3.10, or 3.11