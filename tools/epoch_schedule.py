"""How many barrier epochs does one decode step actually need?

Reads a trace written by `ORNITH_TRACE=<path> ORNITH_PROF=1 ornith`, whose
dispatch lines are `kernel r|w:buffer:allocation:start:end ...`.

A barrier in a Metal compute encoder is encoder-wide: it orders everything
before it against everything after, so the unit that can overlap is the run of
dispatches between two barriers -- an epoch. Two dispatches may share an epoch
only if neither writes bytes the other touches. The fewest epochs any encode
order can achieve is then the longest chain of such conflicts, which is what
levelising as-soon-as-possible gives: a dispatch sits one level below the
deepest thing it collides with, no two dispatches on a level collide (or the
later one would have been pushed down), and every conflict is on an earlier
level. That schedule is reachable by encoding the levels in order.

Three readings of "collide", which is the point of the exercise:

  whole buffer   what the port tracks today: same Metal buffer, either way
                 round. mlx tracks the same thing.
  byte ranges    and the bound ranges overlap, so two views of one allocation
                 that do not share bytes stop counting.
  allocations    and it is the same allocation, so a buffer the pool recycled
                 inside an epoch stops counting. That is ORNITH_NO_RECYCLE=1.

With a kernel-time table as well -- the `GPU time per kernel` block of
`ORNITH_PROF=1 ORNITH_COMMIT_EVERY=1`, saved to a file -- it also costs each
schedule: an epoch is charged its slowest member, on the argument that the
rest runs underneath. That is optimistic (concurrent kernels share the memory
bus, and it ignores the ramp a short read pays), so it under-predicts; check
it against the measured step before trusting it, and read the *ratio* between
two schedules rather than either number.

Run:  python3 tools/epoch_schedule.py <trace> [kernel-times] [dispatches/step]
"""

import sys
from collections import Counter, defaultdict

DEFAULT_STEP = 1669  # dispatches in one decode step of this model


def read(path, n_step):
    """The first step's dispatches, and the epochs the port encoded them in."""
    disp, epochs = [], 1
    for ln in open(path):
        ln = ln.rstrip("\n")
        if ln.startswith("-- "):
            if len(disp) < n_step:
                epochs += 1
            continue
        if len(disp) >= n_step:
            break
        parts = ln.split()
        binds = []
        for b in parts[1:]:
            d, buf, alloc, start, end = b.split(":")
            binds.append((d, buf, int(alloc), int(start), int(end)))
        disp.append((parts[0], binds))
    return disp, epochs


def levelise(disp, key, ranges):
    """ASAP levels, and for each dispatch the one that pushed it down."""
    wrote, read_by = defaultdict(list), defaultdict(list)
    level, pred = [0] * len(disp), [None] * len(disp)
    for i, (_, binds) in enumerate(disp):
        lv, from_ = 1, None
        for (d, buf, alloc, s, e) in binds:
            k = key(buf, alloc)
            # A read waits on earlier writes; a write waits on both.
            against = wrote[k] + read_by[k] if d == "w" else wrote[k]
            for (s2, e2, lv2, i2) in against:
                if (not ranges or (s2 < e and s < e2)) and lv2 >= lv:
                    lv, from_ = lv2 + 1, i2
        level[i], pred[i] = lv, from_
        for (d, buf, alloc, s, e) in binds:
            (wrote if d == "w" else read_by)[key(buf, alloc)].append((s, e, lv, i))
    return level, pred


def short(k):
    return (
        k.replace("_bfloat16", "")
        .replace("_uint32", "")
        .replace("_float32", "")
        .replace("_t_gs_", "/g")
        .replace("_batch_0", "")
    )


# A dispatch in a command buffer of its own pays ~1.5 us for the buffer.
OVERHEAD_MS = 0.0015


def kernel_times(path):
    """`ms per dispatch` per kernel, from a COMMIT_EVERY=1 profile."""
    import re

    per = {}
    for ln in open(path):
        m = re.match(r"\s+([\d.]+) ms\s+[\d.]+%\s+x(\d+)\s+(\S+)", ln)
        if m:
            per[m.group(3)] = max(float(m.group(1)) / int(m.group(2)) - OVERHEAD_MS, 0.0)
    return per


def cost(disp, levels, per):
    """What a schedule costs if every epoch is charged its slowest member."""
    slowest = defaultdict(float)
    for i, lv in enumerate(levels):
        slowest[lv] = max(slowest[lv], per.get(disp[i][0], 0.0))
    return sum(slowest.values())


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "trace.txt"
    times = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else None
    rest = [a for a in sys.argv[2:] if a.isdigit()]
    n_step = int(rest[0]) if rest else DEFAULT_STEP
    disp, encoded = read(path, n_step)
    n = len(disp)
    print(f"{n} dispatches, {encoded} epochs as encoded, {n / encoded:.2f} each\n")
    per = kernel_times(times) if times else {}
    if per:
        _, epochs_now = read(path, n_step)
        levels_now = []
        e = 0
        seen = 0
        for ln in open(path):
            ln = ln.rstrip("\n")
            if ln.startswith("-- "):
                if seen < n_step:
                    e += 1
                continue
            if seen >= n_step:
                break
            levels_now.append(e)
            seen += 1
        base = cost(disp, levels_now, per)
        print(
            f"summed kernel time {sum(per.get(k, 0.0) for k, _ in disp):.2f} ms/token; "
            f"as encoded this schedule costs {base:.2f} ms\n"
        )

    by_buffer = lambda buf, alloc: buf
    by_alloc = lambda buf, alloc: (buf, alloc)
    for name, key, ranges in (
        ("whole buffer (what the port tracks)", by_buffer, False),
        ("byte ranges                        ", by_buffer, True),
        ("allocations (ORNITH_NO_RECYCLE=1)  ", by_alloc, True),
    ):
        level, _ = levelise(disp, key, ranges)
        depth = max(level)
        held = Counter(Counter(level).values())
        priced = f", {cost(disp, level, per):.2f} ms" if per else ""
        print(
            f"{name}: {depth} epochs minimum, {n / depth:.2f} each{priced}"
            f"  (holding 1: {held[1]}, 2: {held[2]}, 3+: {sum(v for k, v in held.items() if k >= 3)})"
        )

    level, pred = levelise(disp, by_alloc, True)
    print(f"\nthe chain that sets the floor, {max(level)} dispatches:")
    i = max(range(n), key=lambda j: level[j])
    chain = []
    while i is not None:
        chain.append(short(disp[i][0]))
        i = pred[i]
    chain.reverse()
    runs, out = [], []
    for k in chain:
        if runs and runs[-1][0] == k:
            runs[-1][1] += 1
        else:
            runs.append([k, 1])
    for k, c in runs[:30]:
        out.append(k if c == 1 else f"{k} x{c}")
    print("  " + " -> ".join(out) + (" -> ..." if len(runs) > 30 else ""))

    groups = defaultdict(list)
    for i, lv in enumerate(level):
        groups[lv].append(short(disp[i][0]))
    together = Counter(
        " + ".join(sorted(ks)) for ks in groups.values() if len(ks) > 1
    )
    print("\nwhat the best order would run in one epoch, most common first:")
    for k, c in together.most_common(10):
        print(f"  x{c:<4} {k}")


main()
