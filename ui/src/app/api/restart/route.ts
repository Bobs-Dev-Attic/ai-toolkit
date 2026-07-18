import { NextResponse } from 'next/server';

/**
 * Stop the Next.js server. The user relaunches manually via
 * Start-AI-Toolkit.bat afterward.
 */
export async function POST() {
  setTimeout(() => {
    console.log('[restart] User requested stop. Exiting process.');
    process.exit(0);
  }, 400);
  return NextResponse.json({ ok: true });
}

export const GET = POST;
