"""Which single dispatch is worth moving, and where to.

`epoch_schedule.py` asks how few barrier epochs a step's dependencies allow,
and prices the answer by charging every epoch its slowest member. It is a
bulk question and it gave a bulk answer -- ~3.4% left in encode order, spread
one epoch at a time, needing a deferred op layer to reach.

That was averaging over dispatches that are not interchangeable. `REBUILD.md`
Phase 31 moved **one** of them -- the shared expert's one-row gate, 12.6 us
alone at the head of an epoch of 6.5 us dispatches, with six epochs of slack
in front of it -- and took 3.3% of a decode step without writing a kernel.
This finds the rest of those by construction rather than by eye.

For every dispatch, as encoded:

  * what its epoch would cost without it (it is only worth moving if it is
    the slowest thing in there, and only by the margin to the second);
  * how far it may move -- after the last dispatch it collides with, before
    the first -- since a barrier is encoder-wide and the tracker keys on the
    buffer;
  * and what the cheapest reachable epoch would cost with it, which is
    nothing at all if that epoch already has something longer in it.

The difference is what moving it saves. This is the same charge-the-slowest
model as the other tool, so it is optimistic in the same ways -- concurrent
kernels share the memory bus -- and it is a *ranking*, not a prediction: it
prices one move at a time, against an epoch structure it assumes stays put.
Check it against the move that has an answer before believing a row of it.

Run:
  ORNITH_INV_DUMP=/tmp/old ORNITH_FUSE_ADD_RMS=0 ORNITH_MOE_EARLY_GATE=1 \
    ORNITH_PROF=1 cargo run --release --bin step_inventory
  python3 tools/epoch_slack.py /tmp/old.trace /tmp/old.tsv
"""

import sys
from collections import defaultdict

# What a dispatch in a command buffer of its own pays for the buffer; the
# times come from ORNITH_COMMIT_EVERY=1 and every one of them carries it.
OVERHEAD_US = 1.5


def read_trace(path, n):
    """`(kernel, binds, epoch)` for the first `n` dispatches."""
    out, epoch = [], 0
    for ln in open(path):
        ln = ln.rstrip("\n")
        if ln.startswith("-- barrier") or ln.startswith("-- end of encoder"):
            if len(out) < n:
                epoch += 1
            continue
        if ln.startswith("--") or ln.startswith("=="):
            continue
        if len(out) >= n:
            break
        parts = ln.split()
        binds = []
        for b in parts[1:]:
            if b.startswith("grid:") or b.startswith("tg:"):
                continue
            d, buf, alloc, s, e = b.split(":")
            binds.append((d, buf, int(s), int(e)))
        out.append((parts[0], binds, epoch))
    return out


def read_times(path):
    """`(section, kernel, us)` per dispatch, in encode order."""
    rows = []
    for ln in open(path):
        if ln.startswith("#"):
            continue
        f = ln.rstrip("\n").split("\t")
        rows.append((f[1], f[2], max(float(f[7]) - OVERHEAD_US, 0.05)))
    return rows


def collisions(disp):
    """For each dispatch, the last one before it and the first after it that
    it may not share an epoch with.

    Two dispatches conflict if they bind the same Metal buffer and at least
    one of them writes it -- mlx's rule, and the port's, and it is what says
    where a dispatch may legally be issued: after everything it collides with
    that is earlier, before everything it collides with that is later.
    Read-read is not a conflict.
    """
    from bisect import bisect_left, bisect_right

    n = len(disp)
    writes, touches = defaultdict(list), defaultdict(list)
    for i, (_, binds, _) in enumerate(disp):
        for (d, buf, _s, _e) in binds:
            touches[buf].append(i)
            if d == "w":
                writes[buf].append(i)

    prev, nxt = [-1] * n, [n] * n
    for i, (_, binds, _) in enumerate(disp):
        for (d, buf, _s, _e) in binds:
            # A write collides with every other access; a read only with
            # writes.
            lst = touches[buf] if d == "w" else writes[buf]
            j = bisect_left(lst, i) - 1
            while j >= 0 and lst[j] == i:
                j -= 1
            if j >= 0:
                prev[i] = max(prev[i], lst[j])
            j = bisect_right(lst, i)
            while j < len(lst) and lst[j] == i:
                j += 1
            if j < len(lst):
                nxt[i] = min(nxt[i], lst[j])
    return prev, nxt


def main():
    trace, times = sys.argv[1], sys.argv[2]
    rows = read_times(times)
    disp = read_trace(trace, len(rows))
    n = min(len(disp), len(rows))
    disp, rows = disp[:n], rows[:n]
    for i in range(n):
        assert disp[i][0] == rows[i][1], f"trace and times disagree at {i}"
    t = [r[2] for r in rows]
    epoch = [d[2] for d in disp]
    n_ep = max(epoch) + 1
    print(f"{n} dispatches in {n_ep} epochs as encoded, {n / n_ep:.2f} each")

    members = defaultdict(list)
    for i in range(n):
        members[epoch[i]].append(i)
    cost = {e: max(t[i] for i in ms) for e, ms in members.items()}
    print(f"charged its slowest member, the encoded schedule is {sum(cost.values()) / 1e3:.3f} ms")

    prev, nxt = collisions(disp)
    # An epoch's cost without one member, so a move can be priced.
    second = {}
    for e, ms in members.items():
        ts = sorted((t[i] for i in ms), reverse=True)
        second[e] = ts[1] if len(ts) > 1 else 0.0

    moves = []
    for i in range(n):
        e = epoch[i]
        if t[i] <= second[e]:
            continue  # not the slowest thing in its epoch: moving it saves nothing
        lo = epoch[prev[i]] + 1 if prev[i] >= 0 else 0
        hi = (epoch[nxt[i]] - 1) if nxt[i] < n else n_ep - 1
        best, best_f = None, None
        for f in range(lo, hi + 1):
            if f == e:
                continue
            add = max(0.0, t[i] - cost.get(f, 0.0))
            if best is None or add < best:
                best, best_f = add, f
        if best is None:
            continue
        gain = t[i] - second[e] - best
        if gain > 0.05:
            moves.append((gain, i, e, best_f, hi - lo))
    moves.sort(reverse=True)

    print(f"\n{len(moves)} dispatches are the slowest in their epoch and have somewhere to go\n")
    by_kernel = defaultdict(lambda: [0, 0.0, 0])
    for gain, i, e, f, span in moves:
        k = by_kernel[rows[i][1]]
        k[0] += 1
        k[1] += gain
        k[2] = max(k[2], span)
    print(f"  {'kernel':<44} {'n':>4} {'us each':>8} {'ms/token':>9} {'slack':>6}")
    for name, (c, g, span) in sorted(by_kernel.items(), key=lambda kv: -kv[1][1])[:18]:
        print(f"  {short(name):<44} {c:>4} {g / c:>8.2f} {g / 1e3:>9.3f} {span:>6}")
    total = sum(g for g, *_ in moves)
    print(f"  {'all of them, if they could all move':<44} {len(moves):>4} {'':>8} {total / 1e3:>9.3f}")

    print("\nthe twenty worst single dispatches, with where they could go:\n")
    print(f"  {'section':<16} {'kernel':<40} {'us':>7} {'epoch':>6} {'->':>5} {'saves':>7}")
    for gain, i, e, f, _ in moves[:20]:
        print(
            f"  {rows[i][0]:<16} {short(rows[i][1]):<40} {t[i]:>7.2f} {e:>6} {f:>5} {gain:>7.2f}"
        )


def short(k):
    for pat in ("_bfloat16", "bfloat16", "_uint32", "_float32"):
        if pat in k:
            return k[: k.index(pat)]
    return k


main()
