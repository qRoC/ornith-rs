//! Byte-pair encoding, a port of `tokenizers`' `models::bpe::Word`.
//!
//! The merge loop is the priority-queue one from `Word::merge_all`, not the
//! textbook "scan for the best pair, repeat": both merge the lowest-ranked
//! pair first, but the queue version fixes the tie-break between two
//! occurrences of the *same* pair (leftmost wins), which matters for
//! overlapping runs like `aaa`.
//!
//! This model's `tokenizer.json` sets `dropout: null`, `unk_token: null`,
//! `continuing_subword_prefix: ""`, `end_of_word_suffix: ""`,
//! `fuse_unk: false`, `byte_fallback: false` and `ignore_merges: false`, so
//! none of the branches those guard are reachable and none are written here.

use std::collections::BinaryHeap;
use std::collections::HashMap;

/// `(rank, new_id)` for every mergeable pair, keyed by the two ids.
pub type Merges = HashMap<(u32, u32), (u32, u32)>;

#[derive(Clone, Copy)]
struct Symbol {
    c: u32,
    prev: i32,
    next: i32,
    /// Characters covered; 0 marks a symbol swallowed by a merge.
    len: usize,
}

/// One entry of the merge queue. `Ord` is reversed on both fields so that
/// `BinaryHeap` pops the lowest rank, and the leftmost position among equal
/// ranks -- `tokenizers`' `impl Ord for Merge`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Merge {
    pos: usize,
    rank: u32,
    new_id: u32,
}

impl Ord for Merge {
    fn cmp(&self, other: &Merge) -> std::cmp::Ordering {
        if self.rank != other.rank {
            other.rank.cmp(&self.rank)
        } else {
            other.pos.cmp(&self.pos)
        }
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Merge) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Merge `ids` (one per character of the byte-level word) all the way down.
pub fn merge_all(ids: &[u32], merges: &Merges) -> Vec<u32> {
    let n = ids.len();
    let mut symbols: Vec<Symbol> = ids
        .iter()
        .enumerate()
        .map(|(i, &c)| Symbol {
            c,
            prev: if i == 0 { -1 } else { i as i32 - 1 },
            next: if i + 1 == n { -1 } else { i as i32 + 1 },
            len: 1,
        })
        .collect();

    let mut queue: BinaryHeap<Merge> = BinaryHeap::with_capacity(n);
    for (pos, w) in symbols.windows(2).enumerate() {
        if let Some(&(rank, new_id)) = merges.get(&(w[0].c, w[1].c)) {
            queue.push(Merge { pos, rank, new_id });
        }
    }

    while let Some(top) = queue.pop() {
        if symbols[top.pos].len == 0 {
            continue;
        }
        if symbols[top.pos].next == -1 {
            continue;
        }
        let next_pos = symbols[top.pos].next as usize;
        let right = symbols[next_pos];

        // A stale queue entry: the pair at `pos` is no longer the one that was
        // pushed.
        match merges.get(&(symbols[top.pos].c, right.c)) {
            Some(&(_, new_id)) if new_id == top.new_id => {}
            _ => continue,
        }

        symbols[top.pos].c = top.new_id;
        symbols[top.pos].len += right.len;
        symbols[top.pos].next = right.next;
        symbols[next_pos].len = 0;
        if right.next > -1 {
            symbols[right.next as usize].prev = top.pos as i32;
        }

        let current = symbols[top.pos];
        if current.prev >= 0 {
            let prev = current.prev as usize;
            if let Some(&(rank, new_id)) = merges.get(&(symbols[prev].c, current.c)) {
                queue.push(Merge { pos: prev, rank, new_id });
            }
        }
        if current.next >= 0 {
            let next = current.next as usize;
            if let Some(&(rank, new_id)) = merges.get(&(current.c, symbols[next].c)) {
                queue.push(Merge { pos: top.pos, rank, new_id });
            }
        }
    }

    symbols.into_iter().filter(|s| s.len != 0).map(|s| s.c).collect()
}
