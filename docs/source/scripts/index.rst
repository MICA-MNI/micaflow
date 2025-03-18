Scripts Reference
===============

This section provides detailed documentation for each command-line utility included in MicaFlow.

Each script can be run independently with the `micaflow [script_name]` command.

Quick Reference
-------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Description
   * - :doc:`SDC`
     - Susceptibility distortion correction
   * - :doc:`apply_SDC`
     - Apply precomputed distortion correction
   * - :doc:`apply_warp`
     - Apply transformations to images
   * - :doc:`bet`
     - Brain extraction using HD-BET
   * - :doc:`bias_correction`
     - N4 bias field correction
   * - :doc:`calculate_jaccard`
     - Calculate similarity between segmentations
   * - :doc:`compute_fa_md`
     - Compute DTI metrics (FA, MD)
   * - :doc:`coregister`
     - Image coregistration using ANTs
   * - :doc:`denoise`
     - Denoise diffusion-weighted images
   * - :doc:`motion_correction`
     - Motion correction for DWI
   * - :doc:`synthseg`
     - Deep learning segmentation with SynthSeg
   * - :doc:`texture_generation`
     - Generate texture features


.. toctree::
   :maxdepth: 1
   :hidden:

   SDC
   apply_SDC
   apply_warp
   bet
   bias_correction
   calculate_jaccard
   compute_fa_md
   coregister
   denoise
   motion_correction
   synthseg
   texture_generation
