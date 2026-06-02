Getting Started
===============

Installation
------------

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"

Command-line usage
------------------

.. code-block:: bash

   fg-vector-sampler --xyz monomer/opt-PT-B97-3c.xyz monomer/opt-HNO3-B97-3c.xyz \
     --names PT HNO3 --output outputs/sampling --max-candidates 100 --beam-width 50

Recommended workflow
--------------------

1. Prepare optimized monomer geometries in ``monomer/``.
2. Verify monomer feature generation.
3. Run a small control sampling job.
4. Scale out with parallel jobs once the parameter window is validated.
5. Archive outputs together with the config and changelog entry.
