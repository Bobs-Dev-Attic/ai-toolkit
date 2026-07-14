import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { getTrainingFolder, flushCache } from '@/server/settings';
import { HF_CACHE_KEY, dirSizeBytes, diskFor, nearestExistingDir } from '@/server/hfCache';

export const runtime = 'nodejs';

const prisma = new PrismaClient();

interface MoveStatus {
  state: 'running' | 'done' | 'error' | 'idle';
  source: string;
  dest: string;
  sourceBytes: number;
  destBytes?: number;
  pid?: number | null;
  startedAt?: number;
  error?: string;
}

async function statusFilePath(): Promise<string> {
  const trainingFolder = await getTrainingFolder();
  return path.join(trainingFolder, '.hf_cache_move.json');
}

function readStatus(p: string): MoveStatus | null {
  try {
    return JSON.parse(fs.readFileSync(p, 'utf-8')) as MoveStatus;
  } catch {
    return null;
  }
}

function writeStatus(p: string, s: MoveStatus) {
  try {
    fs.writeFileSync(p, JSON.stringify(s, null, 2));
  } catch {
    /* best effort */
  }
}

function pidAlive(pid: number | null | undefined): boolean {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (e: any) {
    return e?.code === 'EPERM'; // exists but not ours
  }
}

// GET -> current move status (advances running->done/error when the process exits)
export async function GET() {
  const p = await statusFilePath();
  const s = readStatus(p);
  if (!s) return NextResponse.json({ state: 'idle' } as MoveStatus);

  if (s.state === 'running') {
    const destBytes = dirSizeBytes(s.dest);
    if (pidAlive(s.pid)) {
      return NextResponse.json({ ...s, destBytes });
    }
    // process gone — decide success/failure by how much made it across
    const ok = s.sourceBytes === 0 || destBytes >= s.sourceBytes * 0.98;
    const next: MoveStatus = { ...s, destBytes, state: ok ? 'done' : 'error', pid: null };
    if (!ok) next.error = 'Move ended before all data was copied. The source cache was left intact.';
    writeStatus(p, next);
    return NextResponse.json(next);
  }

  return NextResponse.json({ ...s, destBytes: dirSizeBytes(s.dest) });
}

// POST { source, dest } -> start a move (or do an instant same-volume rename)
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const source: string = String(body.source || '');
    const dest: string = String(body.dest || '');

    if (!source || !dest) {
      return NextResponse.json({ error: 'source and dest are required' }, { status: 400 });
    }
    const src = path.resolve(source);
    const dst = path.resolve(dest);
    if (src === dst) {
      return NextResponse.json({ error: 'Source and destination are the same.' }, { status: 400 });
    }
    if (dst.toLowerCase().startsWith(src.toLowerCase() + path.sep)) {
      return NextResponse.json({ error: 'Destination is inside the source folder.' }, { status: 400 });
    }
    if (!fs.existsSync(src)) {
      return NextResponse.json({ error: 'Current cache folder does not exist — nothing to move.' }, { status: 400 });
    }

    // Refuse while training is active — the cache may be in use.
    const running = await prisma.job.count({ where: { status: 'running' } });
    if (running > 0) {
      return NextResponse.json(
        { error: 'A training job is running. Stop it before moving the cache.' },
        { status: 409 },
      );
    }

    const p = await statusFilePath();
    const existing = readStatus(p);
    if (existing?.state === 'running' && pidAlive(existing.pid)) {
      return NextResponse.json({ error: 'A move is already in progress.' }, { status: 409 });
    }

    const sourceBytes = dirSizeBytes(src);

    // Ensure destination parent exists.
    fs.mkdirSync(path.dirname(dst), { recursive: true });

    // Space check for a cross-volume copy.
    const destDisk = await diskFor(dst);
    const sameNearest =
      nearestExistingDir(src).slice(0, 2).toLowerCase() === nearestExistingDir(dst).slice(0, 2).toLowerCase();
    if (!sameNearest && destDisk && destDisk.freeBytes < sourceBytes) {
      return NextResponse.json(
        { error: `Not enough space: need ${(sourceBytes / 1024 ** 3).toFixed(0)} GB, have ${(destDisk.freeBytes / 1024 ** 3).toFixed(0)} GB free.` },
        { status: 400 },
      );
    }

    // Persist the setting so the location and the data agree from here on.
    await prisma.settings.upsert({
      where: { key: HF_CACHE_KEY },
      update: { value: dst },
      create: { key: HF_CACHE_KEY, value: dst },
    });
    flushCache();

    // Fast path: same volume + dest doesn't exist -> instant rename.
    if (!fs.existsSync(dst)) {
      try {
        fs.renameSync(src, dst);
        const done: MoveStatus = { state: 'done', source: src, dest: dst, sourceBytes, destBytes: sourceBytes, pid: null };
        writeStatus(p, done);
        return NextResponse.json({ ...done, instant: true });
      } catch (e: any) {
        // EXDEV (cross-device) or dest-exists: fall through to a copy.
        if (e?.code !== 'EXDEV' && e?.code !== 'ENOTEMPTY' && e?.code !== 'EEXIST' && e?.code !== 'EPERM') {
          // Unexpected — still try the copy path below.
          console.error('rename failed, falling back to copy:', e);
        }
      }
    }

    // Background copy. robocopy on Windows, cp+rm elsewhere.
    const logPath = path.join(path.dirname(p), '.hf_cache_move.log');
    const logFd = fs.openSync(logPath, 'a');
    let child;
    if (process.platform === 'win32') {
      child = spawn(
        'robocopy',
        [src, dst, '/E', '/MOVE', '/R:1', '/W:1', '/NP', '/NFL', '/NDL', '/NJH', '/NJS', '/MT:16'],
        { detached: true, stdio: ['ignore', logFd, logFd], windowsHide: true },
      );
    } else {
      child = spawn('sh', ['-c', `cp -a "${src}/." "${dst}/" && rm -rf "${src}"`], {
        detached: true,
        stdio: ['ignore', logFd, logFd],
      });
    }
    child.unref();

    const status: MoveStatus = {
      state: 'running',
      source: src,
      dest: dst,
      sourceBytes,
      destBytes: 0,
      pid: child.pid ?? null,
      startedAt: Date.now(),
    };
    writeStatus(p, status);
    return NextResponse.json(status);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to start move: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 },
    );
  }
}
