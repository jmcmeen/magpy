"""
MagPy -- a PyQt6 GUI for bioacoustics analysis.

MagPy is a thin GUI wrapper around the ``bioamla`` library. bioamla owns the
domain (audio I/O, transforms, spectrograms, indices, detection, annotations,
ML, clustering, catalogs, playback); MagPy owns the interactive UI and the
mutable application state that a functional library cannot hold for us.

Layering (see ARCHITECTURE.md):
    services/  thin wrappers over bioamla; the only layer that imports bioamla
    workers/   QThreadPool bridge for long-running bioamla calls
    models/    mutable app state (Document, Selection/Annotation), Qt signals
    widgets/   dumb Qt views; never import bioamla
    screens/   compose widgets + view-models into workspace views
"""

__version__ = "0.2.0"
__author__ = "John McMeen"
