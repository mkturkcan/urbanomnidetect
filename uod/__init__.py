"""UrbanOmniDetect inference / real-time BEV runtime package.

Modules
-------
keypoints   parse pose-model outputs; resolve ground-corner indices
aux_head    ridge map from auxiliary boxes to ground centres (paper Sec. 3.5)
bev         temporally stable radar-style BEV viewport + rendering
tracking    real-time temporal tracker with motion-coasted persistence
viz         drawing helpers for the source frame and BEV
model       model loading + optional TensorRT export

The orthogonality homography solver lives in the top-level module
``homography_rt`` (named to match the repository structure).
"""

__version__ = "0.1.0"
