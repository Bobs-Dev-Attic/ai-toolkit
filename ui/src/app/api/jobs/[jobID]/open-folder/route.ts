import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import { getTrainingFolder } from '@/server/settings';

export const runtime = 'nodejs';

const prisma = new PrismaClient();

// Opens the given folder in the host OS file manager (Explorer on Windows,
// Finder on macOS, the default file manager on Linux). Only usable when the
// UI server runs on the same machine as the user (the normal local setup).
function openInFileManager(folderPath: string) {
  if (process.platform === 'win32') {
    // explorer.exe frequently exits with code 1 even on success, so we don't
    // treat the exit code as an error. Detach so it outlives this request.
    spawn('explorer.exe', [folderPath], { detached: true, stdio: 'ignore' }).unref();
  } else if (process.platform === 'darwin') {
    spawn('open', [folderPath], { detached: true, stdio: 'ignore' }).unref();
  } else {
    spawn('xdg-open', [folderPath], { detached: true, stdio: 'ignore' }).unref();
  }
}

export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const samplesFolder = path.join(jobFolder, 'samples');

  // Prefer the samples folder, but fall back to the job folder when no samples
  // have been generated yet (sampling disabled, or training just started).
  // Previously this 404'd and nothing opened, which read as "Open Folder is broken".
  let target: string | null = null;
  if (fs.existsSync(samplesFolder)) {
    target = samplesFolder;
  } else if (fs.existsSync(jobFolder)) {
    target = jobFolder;
  }

  if (!target) {
    return NextResponse.json({ error: 'Job folder does not exist yet' }, { status: 404 });
  }

  try {
    openInFileManager(target);
    return NextResponse.json({ ok: true, path: target });
  } catch (err) {
    console.error('Error opening folder:', err);
    return NextResponse.json({ error: 'Failed to open folder' }, { status: 500 });
  }
}
