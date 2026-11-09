"""A small synthetic score curve; no video/model downloads required."""
import numpy as np
from aps import APSConfig, select_frames

scores = np.r_[np.linspace(.1,.2,20), np.linspace(.3,.95,20), np.linspace(.8,.2,20)]
frames = list(range(0,len(scores)*30,30))
selected, trace = select_frames(scores,frames,APSConfig(budget=8,beta=.6),return_details=True)
print('Selected original video frames:', selected)
print('Half-open candidate intervals:', trace['intervals'])
print('Frames per interval:', trace['allocation'])
