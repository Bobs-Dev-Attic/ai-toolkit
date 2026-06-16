import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawn } from 'child_process';
import { getDatasetsRoot } from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';

const imageExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp']);

export type ImageOpScript =
  | 'caption_dataset.py'
  | 'resize_images.py'
  | 'remove_background.py'
  | 'upscale_images.py'
  | 'auto_crop.py';

export interface SpawnImageOpArgs {
  request: Request;
  /** Python script file under <toolkit>/scripts to spawn. */
  script: ImageOpScript;
  /** Image paths the operation should process. Required for non-caption ops; optional for caption (full-dataset mode). */
  images: string[];
  /** Extra CLI args appended after --image-list. */
  extraArgs?: string[];
  /** When true, do not pass --image-list; pass --dataset-dir + datasetName resolution instead. */
  skipImageList?: boolean;
  /** Dataset folder (already resolved + validated). Used to constrain image paths to its tree. */
  datasetFolder?: string;
}

export function isSubPath(parent: string, child: string): boolean {
  const relative = path.relative(parent, child);
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

export function getPythonExecutable(): string {
  const embeddedPython = path.resolve(TOOLKIT_ROOT, '..', 'python_embeded', 'python.exe');
  if (fs.existsSync(embeddedPython)) return embeddedPython;
  return process.platform === 'win32' ? 'python' : 'python3';
}

export async function resolveDatasetFolder(datasetName: string): Promise<string> {
  const datasetsPath = path.resolve(await getDatasetsRoot());
  const datasetFolder = path.resolve(datasetsPath, datasetName);
  if (!isSubPath(datasetsPath, datasetFolder)) {
    throw new Error('Invalid dataset name');
  }
  if (!fs.existsSync(datasetFolder)) {
    throw new Error(`Folder '${datasetName}' not found`);
  }
  return datasetFolder;
}

/**
 * Validate a list of image paths against a dataset folder. Returns the
 * filtered list of valid absolute paths.
 */
export function filterImagesWithinFolder(images: unknown, datasetFolder: string): string[] {
  if (!Array.isArray(images)) return [];
  return images.filter(
    (imgPath): imgPath is string =>
      typeof imgPath === 'string' &&
      imageExtensions.has(path.extname(imgPath).toLowerCase()) &&
      isSubPath(datasetFolder, path.resolve(imgPath)) &&
      fs.existsSync(imgPath),
  );
}

/**
 * Spawn a Python script that emits newline-delimited JSON events on stdout
 * and stream them back to the client as Server-Sent Events.
 *
 * Centralizes: temp image-list file lifecycle, stderr passthrough, error
 * fallback, client-abort handling.
 */
export function spawnImageOpStream(opts: SpawnImageOpArgs): Response {
  const { request, script, images, extraArgs = [], skipImageList, datasetFolder } = opts;

  const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', script);
  const scriptArgs: string[] = [scriptPath];
  if (datasetFolder) {
    scriptArgs.push('--dataset-dir', datasetFolder);
  }

  let tempListPath: string | null = null;
  if (!skipImageList) {
    if (images.length === 0) {
      return new Response(JSON.stringify({ error: 'No valid images provided' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    tempListPath = path.join(os.tmpdir(), `aitk-imgop-${Date.now()}-${Math.round(Math.random() * 1e9)}.txt`);
    fs.writeFileSync(tempListPath, images.join('\n'), 'utf-8');
    scriptArgs.push('--image-list', tempListPath);
  }
  scriptArgs.push(...extraArgs);

  const child = spawn(getPythonExecutable(), scriptArgs, {
    cwd: TOOLKIT_ROOT,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' },
    windowsHide: true,
  });

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    start(controller) {
      let closed = false;
      let stdoutBuffer = '';
      let stderrTail = '';
      let scriptReportedError = false;

      const cleanupTempFile = () => {
        if (tempListPath) {
          try {
            fs.unlinkSync(tempListPath);
          } catch {
            // best effort
          }
          tempListPath = null;
        }
      };

      const send = (event: object) => {
        if (closed) return;
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      };

      const finish = () => {
        if (closed) return;
        closed = true;
        cleanupTempFile();
        try {
          controller.close();
        } catch {
          // already closed
        }
      };

      child.stdout.on('data', (data: Buffer) => {
        stdoutBuffer += data.toString();
        let newlineIndex: number;
        while ((newlineIndex = stdoutBuffer.indexOf('\n')) >= 0) {
          const line = stdoutBuffer.slice(0, newlineIndex).trim();
          stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);
          if (!line) continue;
          try {
            const parsed = JSON.parse(line);
            if (parsed && parsed.type === 'error') scriptReportedError = true;
            send(parsed);
          } catch {
            // ignore non-JSON noise
          }
        }
      });

      child.stderr.on('data', (data: Buffer) => {
        const text = data.toString();
        process.stderr.write(`[${script}] ${text}`);
        stderrTail = (stderrTail + text).slice(-4000);
      });

      child.on('error', err => {
        send({ type: 'error', message: err.message });
        finish();
      });

      child.on('close', code => {
        const remaining = stdoutBuffer.trim();
        if (remaining) {
          try {
            const parsed = JSON.parse(remaining);
            if (parsed && parsed.type === 'error') scriptReportedError = true;
            send(parsed);
          } catch {
            // ignore
          }
        }
        if (code !== 0 && !scriptReportedError) {
          const filteredLines = stderrTail
            .split(/\r?\n/)
            .map(l => l.trim())
            .filter(line => line && !/NOTE: Redirects are currently not supported/i.test(line))
            .slice(-6);
          const message = filteredLines.length ? filteredLines.join('\n') : `${script} exited with code ${code}`;
          send({ type: 'error', message });
        }
        send({ type: 'end' });
        finish();
      });

      request.signal.addEventListener('abort', () => {
        try {
          child.kill();
        } catch {
          // already gone
        }
        finish();
      });
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}
