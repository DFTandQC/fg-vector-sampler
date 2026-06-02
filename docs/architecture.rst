Architecture
============

Core modules
------------

* ``fg_vector_sampler.core`` — atoms, features, contacts, and geometry helpers
* ``fg_vector_sampler.features`` — feature generation from monomer templates
* ``fg_vector_sampler.priors`` — contact templates and scoring priors
* ``fg_vector_sampler.sampler`` — beam-search cluster generation engine
* ``fg_vector_sampler.analysis`` — post-processing and ensemble statistics
* ``fg_vector_sampler.io`` — XYZ / JSON input and output helpers
* ``fg_vector_sampler.cli`` — command-line interface

Sampling pipeline
-----------------

#. Load monomer structures
#. Shift monomers to their center of mass
#. Generate functional-group features
#. Enumerate compatible contact modes
#. Grow a beam of candidate clusters
#. Filter, deduplicate, and export the final ensemble

Performance notes
-----------------

The beam search has quadratic-like growth in the number of retained candidates.
For research-scale runs, prefer multiple moderate jobs over one extremely large beam.
