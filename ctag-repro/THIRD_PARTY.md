# Third-party provenance

This implementation derives method details and small portions of control-flow
behavior from PapayaResearch/ctag, MIT licensed, copyright 2024 Manuel Cherep
and Nikhil Singh. The pinned source revision is
`fc207b271a9761a6b001e3d028e777d608c4e91f`.
The ESC-10, ESC-50, and AudioSet prompt lists in `data/` are copied from that
revision under the same license.

SynthAX 0.2.1 (legacy reference) and 0.2.2 (modern Colab adapter) are MIT
licensed. LAION-CLAP code and checkpoints are external dependencies and are
not redistributed by this repository. See each upstream project for its
applicable code, data, and model terms.

The optional direct-model training workflow downloads the 527-class AudioSet
label vocabulary from Google's archived `audioset/ontology` release. The
AudioSet ontology is licensed CC BY-SA 4.0; only display names are used as
training prompts and the downloaded CSV is stored in the user's workspace.
