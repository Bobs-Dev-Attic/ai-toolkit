// Naming helper for clone/duplicate actions.
//
// When duplicating a job or dataset we want a sensible next name instead of an
// ever-growing "_copy" chain. Convention: names version themselves with a
// trailing "_v<number>" (e.g. katiedarling_krea2_v0), so cloning should bump
// that number rather than append "_copy". Names without a version suffix fall
// back to "_copy".
//
// When `taken` (existing names) is provided, the result is guaranteed free:
//   - versioned names keep incrementing the version until one is unused
//   - unversioned names try _copy, then _copy2, _copy3, …
// Without `taken` it returns the first candidate and leaves collision handling
// to the caller (e.g. the save endpoint's duplicate-name check).

const VERSION_RE = /^(.*)_v(\d+)$/;

export function nextCloneName(name: string, taken?: Iterable<string>): string {
  const takenSet = taken ? new Set(taken) : null;
  const m = VERSION_RE.exec(name);

  if (m) {
    const base = m[1];
    let n = parseInt(m[2], 10) + 1;
    let candidate = `${base}_v${n}`;
    if (takenSet) {
      while (takenSet.has(candidate)) {
        n += 1;
        candidate = `${base}_v${n}`;
      }
    }
    return candidate;
  }

  let candidate = `${name}_copy`;
  if (takenSet) {
    let i = 2;
    while (takenSet.has(candidate)) {
      candidate = `${name}_copy${i}`;
      i += 1;
    }
  }
  return candidate;
}
