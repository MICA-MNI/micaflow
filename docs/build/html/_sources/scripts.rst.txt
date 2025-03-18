Scripts
=======

This section provides documentation for the individual command-line scripts included in MicaFlow.

Each script can be run independently using the ``micaflow [command]`` syntax and has its own set of parameters.

Quick Reference
-------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Description
   * - :doc:`scripts/apply_SDC`
     - Apply precomputed distortion correction
   * - :doc:`scripts/apply_warp`
     - Apply transformations to images
   * - :doc:`scripts/bet`
     - Brain extraction using HD-BET
   * - :doc:`scripts/bias_correction`
     - N4 bias field correction
   * - :doc:`scripts/calculate_jaccard`
     - Calculate similarity between segmentations
   * - :doc:`scripts/compute_fa_md`
     - Compute DTI metrics (FA, MD)
   * - :doc:`scripts/coregister`
     - Image coregistration using ANTs
   * - :doc:`scripts/denoise`
     - Denoise diffusion-weighted images
   * - :doc:`scripts/motion_correction`
     - Motion correction for DWI
   * - :doc:`scripts/SDC`
     - Susceptibility distortion correction
   * - :doc:`scripts/synthseg`
     - Deep learning segmentation with SynthSeg
   * - :doc:`scripts/texture_generation`
     - Generate texture features

.. toctree::
   :maxdepth: 1
   :hidden:

   scripts/apply_SDC
   scripts/apply_warp
   scripts/bet
   scripts/bias_correction
   scripts/calculate_jaccard
   scripts/compute_fa_md
   scripts/coregister
   scripts/denoise
   scripts/motion_correction
   scripts/SDC
   scripts/synthseg
   scripts/texture_generation