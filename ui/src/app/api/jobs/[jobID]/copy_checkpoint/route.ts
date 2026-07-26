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

  // Main checkpoints are named `{job.name}_{step:09d}.safetensors`. Exclude
  // auxiliary safetensors (e.g. CRITIC_*) that can share the folder.
  const stepOf = (name: string): number => {
    const m = name.match(/_(\d+)\.safetensors$/);
    return m ? parseInt(m[1], 10) : -1;
  };
  const candidates = entries.filter(
    f => f.endsWith('.safetensors') && !f.startsWith('CRITIC_'),
  );
  if (candidates.length === 0) {
    return NextResponse.json({ error: 'No checkpoint found for this job yet' }, { status: 404 });
  }

  // Highest training step wins; mtime breaks ties / unparseable names.
  let latest = candidates[0];
  let latestKey: [number, number] = [-1, -1];
  for (const f of candidates) {
    let mtime = 0;
    try {
      mtime = (await fs.promises.stat(path.join(jobFolder, f))).mtimeMs;
    } catch {
      // skip unreadable entries
      continue;
    }
    const key: [number, number] = [stepOf(f), mtime];
    if (key[0] > latestKey[0] || (key[0] === latestKey[0] && key[1] > latestKey[1])) {
      latest = f;
      latestKey = key;
    }
  }

  const from = path.join(jobFolder, latest);
  const to = path.join(dest, latest);
  try {
    await fs.promises.mkdir(dest, { recursive: true });
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
