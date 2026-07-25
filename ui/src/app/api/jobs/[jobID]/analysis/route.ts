import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import sqlite3 from 'sqlite3';
import { getTrainingFolder } from '@/server/settings';
import { analyzeRun, RunPoint } from '@/utils/runAnalysis';
import { JobConfig } from '@/types';

export const runtime = 'nodejs';

const prisma = new PrismaClient();

function openDb(filename: string) {
  const db = new sqlite3.Database(filename, sqlite3.OPEN_READONLY);
  db.configure('busyTimeout', 30_000);
  return db;
}
function all<T = any>(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T[]>((resolve, reject) => {
    db.all(sql, params, (err, rows) => (err ? reject(err) : resolve(rows as T[])));
  });
}
function closeDb(db: sqlite3.Database) {
  return new Promise<void>(resolve => db.close(() => resolve()));
}

// Read loss values (in step order) from loss_log.db, tolerating either the
// 'loss/loss' or legacy 'loss' metric key. Returns [] on any failure — the
// post-mortem degrades gracefully without loss data.
async function readLoss(logPath: string): Promise<number[]> {
  if (!fs.existsSync(logPath)) return [];
  const db = openDb(logPath);
  try {
    const keys = await all<{ key: string }>(db, `SELECT key FROM metric_keys`);
    const names = keys.map(k => k.key);
    const lossKey = names.includes('loss/loss') ? 'loss/loss' : names.includes('loss') ? 'loss' : null;
    if (!lossKey) return [];
    const rows = await all<{ value: number | null; value_text: string | null }>(
      db,
      `SELECT value_real AS value, value_text FROM metrics WHERE key = ? ORDER BY step ASC`,
      [lossKey],
    );
    return rows
      .map(r => (r.value != null ? r.value : r.value_text ? Number(r.value_text) : NaN))
      .filter(v => typeof v === 'number' && !Number.isNaN(v));
  } catch {
    return [];
  } finally {
    await closeDb(db);
  }
}

function readSystemStats(logPath: string): RunPoint[] {
  if (!fs.existsSync(logPath)) return [];
  let content = '';
  try {
    content = fs.readFileSync(logPath, 'utf-8');
  } catch {
    return [];
  }
  const points: RunPoint[] = [];
  for (const line of content.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try {
      points.push(JSON.parse(s));
    } catch {
      // skip a partially-written trailing line
    }
  }
  return points;
}

export async function GET(_request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);

  const points = readSystemStats(path.join(jobFolder, 'system_stats.jsonl'));
  const loss = await readLoss(path.join(jobFolder, 'loss_log.db'));

  let config: JobConfig | null = null;
  try {
    config = JSON.parse(job.job_config);
  } catch {
    config = null;
  }

  const analysis = analyzeRun(points, loss, config);
  return NextResponse.json(analysis);
}
