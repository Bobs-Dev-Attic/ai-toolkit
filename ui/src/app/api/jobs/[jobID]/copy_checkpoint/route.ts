import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

export const runtime = 'nodejs';

// Copies the job's latest (highest-step) .safetensors checkpoint to a
// user-chosen folder. This is the on-demand counterpart to the config-driven
// copy_final_to that the Python trainer performs on completion.
export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  let dest = '';
  try {
    const body = await request.json();
    dest = (body?.dest ?? '').toString().trim();
  } catch {
    // no/invalid body handled below
  }
  if (!dest) {
    return NextResponse.json({ error: 'A destination folder is required' }, { status: 400 });
  }

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);

  let entries: string[];
  try {
    entries = await fs.promises.readdir(jobFolder);
  } catch {
    return NextResponse.json({ error: 'Job folder does not exist yet' }, { status: 404 });
  }

  // Periodic checkpoints are named `{job.name}_{step:09d}.safetensors`; the
  // end-of-run save writes an unnumbered `{job.name}.safetensors` — the true
  // final checkpoint. `stepOf` returns null for that unnumbered file so it can
  // be ranked above every numbered save. Exclude auxiliary safetensors
  // (e.g. CRITIC_*) that can share the folder.
  const stepOf = (name: string): number | null => {
    const m = name.match(/_(\d+)\.safetensors$/);
    return m ? parseInt(m[1], 10) : null;
  };
  const candidates = entries.filter(
    f => f.endsWith('.safetensors') && !f.startsWith('CRITIC_'),
  );
  if (candidates.length === 0) {
    return NextResponse.json({ error: 'No checkpoint found for this job yet' }, { status: 404 });
  }

  // The unnumbered final wins over every numbered save; among numbered saves
  // the highest step wins; mtime breaks any remaining ties.
  let latest = candidates[0];
  let latestKey: [number, number, number] = [-1, -1, -1];
  for (const f of candidates) {
    let mtime = 0;
    try {
      mtime = (await fs.promises.stat(path.join(jobFolder, f))).mtimeMs;
    } catch {
      // skip unreadable entries
      continue;
    }
    const step = stepOf(f);
    const key: [number, number, number] = [step === null ? 1 : 0, step ?? -1, mtime];
    if (
      key[0] > latestKey[0] ||
      (key[0] === latestKey[0] && key[1] > latestKey[1]) ||
      (key[0] === latestKey[0] && key[1] === latestKey[1] && key[2] > latestKey[2])
    ) {
      latest = f;
      latestKey = key;
    }
  }

  const from = path.join(jobFolder, latest);
  try {
    await fs.promises.mkdir(dest, { recursive: true });
    let to = path.join(dest, latest);
    // Never overwrite: on a name collision, insert a timestamp before the ext.
    try {
      await fs.promises.access(to);
      const ext = path.extname(latest);
      const stem = path.basename(latest, ext);
      const d = new Date();
      const p = (n: number) => String(n).padStart(2, '0');
      const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
      to = path.join(dest, `${stem}_${stamp}${ext}`);
    } catch {
      // no existing file at that name — use it as-is
    }
    await fs.promises.copyFile(from, to);
    return NextResponse.json({ ok: true, from, to });
  } catch (err: any) {
    console.error('Error copying checkpoint:', err);
    return NextResponse.json(
      { error: `Failed to copy checkpoint: ${err?.message ?? 'unknown error'}` },
      { status: 500 },
    );
  }
}
