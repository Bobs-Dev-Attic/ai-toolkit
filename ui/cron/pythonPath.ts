import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT } from './paths';

const isWindows = process.platform === 'win32';

// Shared resolver used by both the cron worker and Next.js API routes
// so the Python interpreter is configured in exactly one place.
export const resolvePythonPath = (): string => {
  const candidates: string[] = [];

  if (isWindows) {
    // Bundled python_embeded ships next to AI-Toolkit in the Easy-Install layout
    // (one or two directories above TOOLKIT_ROOT). Prefer it over .venv/venv
    // and over whatever bare `python.exe` is on PATH, because this is the
    // interpreter that has the toolkit's pinned packages installed.
    candidates.push(path.resolve(TOOLKIT_ROOT, '..', 'python_embeded', 'python.exe'));
    candidates.push(path.resolve(TOOLKIT_ROOT, '..', '..', 'python_embeded', 'python.exe'));
    candidates.push(path.resolve(TOOLKIT_ROOT, '..', '..', '..', 'python_embeded', 'python.exe'));
    candidates.push(path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe'));
    candidates.push(path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe'));
  } else {
    candidates.push(path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python'));
    candidates.push(path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python'));
  }

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return isWindows ? 'python.exe' : 'python3';
};
