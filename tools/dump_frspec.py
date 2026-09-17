"""Write the FR-Spec ranked sub-vocabulary this port drafts from.

FR-Spec prunes the *draft* head to a frequency-ranked slice of the vocabulary
while the target keeps verifying over all of it. The emitted tokens cannot
change -- every committed token is the verify forward's own argmax, and the
draft only decides how many of them one cycle gets -- so this is a pure speed
lever whose whole cost is acceptance.

The ranked list is MTPLX's, `mtplx/data/qwen38_code_ranked_64k.npy`: 65536 ids
chosen by frequency over a generic corpus (deliberately not workload-fit) and
then sorted, which is why it reads as ascending rather than ranked. Its own
receipts are 0.99487 coverage out-of-sample at n = 65536 and 0.99728 on real
traces, for an acceptance-length cost of <= 2.2%. It is the same tokenizer as
this artifact -- 248320 rows -- so it loads unchanged.

It is not copied into this repository, for the same reason `refs/` is not: it
is somebody else's artifact and this is its writer.

    PYTHONPATH=../../vendor/MTPLX python3 dump_frspec.py
"""

import pathlib
import struct
import sys

import numpy as np

VOCAB = 248320
SRC = pathlib.Path("../../vendor/MTPLX/mtplx/data/qwen38_code_ranked_64k.npy")
OUT = pathlib.Path("../refs/frspec/ids_64k.bin")


def main() -> None:
    src = SRC.resolve()
    if not src.exists():
        sys.exit(f"{src} is missing; MTPLX must be vendored next to this port")
    ids = np.load(src, allow_pickle=False).reshape(-1)

    # The same checks `load_frspec_ids` and `install_frspec_draft_head` make,
    # because a bad list here is a wrong token rather than a slow one.
    n = int(ids.shape[0])
    assert n > 0, "no ids"
    assert n < VOCAB, f"{n} is not actually pruned against {VOCAB}"
    assert int(ids.min()) >= 0 and int(ids.max()) < VOCAB, "ids out of range"
    assert len(np.unique(ids)) == n, "duplicate ids"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(ids.astype("<u4").tobytes())
    print(f"{OUT.resolve()}: {n} ids, {n / VOCAB:.4f} of the vocabulary")


main()
