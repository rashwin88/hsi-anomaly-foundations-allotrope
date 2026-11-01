"""
The story GIF: one looping animation of how a scene was narrowed down to the selected anomalies.

  render.py   frames (image, caption, overlay) -> GIF bytes; numpy + PIL only
  chain.py    what a story is told from: the chain's directories, configs, summaries; readers
  frames.py   a spectral_library_match chain's output files -> frames

Served by GET /actions/{id}/story.gif (backend/allotrope/api/action_story.py); design in
docs/lld/story-gif.md. A presentation artefact only: it contains a heatmap frame, so it is never
part of a submission bundle.
"""
