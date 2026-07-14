import { NextResponse } from 'next/server';
import fs from 'fs';
import { getTrainingFolder, getDatasetsRoot } from '@/server/settings';

export const runtime = 'nodejs';

interface DiskInfo {
  path: string;
  totalBytes: number;
  freeBytes: number;
}

async function diskFor(p: string): Promise<DiskInfo | null> {
  try {
    // fs.statfs is available in Node 18.15+ / 20+.
    const st: any = await (fs.promises as any).statfs(p);
    const totalBytes = st.blocks * st.bsize;
    const freeBytes = st.bavail * st.bsize;
    return { path: p, totalBytes, freeBytes };
  } catch (err) {
    console.error('statfs failed for', p, err);
    return null;
  }
}

export async function GET() {
  try {
    const trainingFolder = await getTrainingFolder();
    let datasetsRoot = '';
    try {
      datasetsRoot = await getDatasetsRoot();
    } catch {
      /* optional */
    }

    const training = await diskFor(trainingFolder);
    const datasets = datasetsRoot ? await diskFor(datasetsRoot) : null;

    return NextResponse.json({ training, datasets });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to read disk info: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 },
    );
  }
}
