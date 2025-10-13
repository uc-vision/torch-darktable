import numpy as np

from torch_darktable.debayer import BayerPattern

# Pattern mapping to [R, G1, G2, B] position indices
pattern_map = {
  BayerPattern.RGGB: (0, 1, 2, 3),  # R=TL, G=TR+BL, B=BR
  BayerPattern.BGGR: (3, 1, 2, 0),  # B=TL, G=TR+BL, R=BR
  BayerPattern.GRBG: (1, 0, 3, 2),  # G=TL+BR, R=TR, B=BL
  BayerPattern.GBRG: (2, 0, 3, 1),  # G=TL+BR, B=TR, R=BL
}


def extract_bayer_channels(bayer_data, pattern: BayerPattern):
  positions = [bayer_data[i::2, j::2] for i in (0, 0, 1, 1) for j in (0, 1, 0, 1)]
  r, g1, g2, b = pattern_map[pattern]
  return (
    positions[r].flatten(),
    np.concatenate([positions[g1].flatten(), positions[g2].flatten()]),
    positions[b].flatten(),
  )


def get_channel_statistics(bayer_data, pattern: BayerPattern):
  r_channel, g_channel, b_channel = extract_bayer_channels(bayer_data, pattern)
  return (
    float(np.mean(r_channel)),
    float(np.mean(g_channel)),
    float(np.mean(b_channel)),
    float(np.std(r_channel)),
    float(np.std(g_channel)),
    float(np.std(b_channel)),
  )
