import { NextResponse } from 'next/server';
import fs from 'fs';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

const ALLOWED_EXT = /\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav)$/i;

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPaths } = body;
    if (!Array.isArray(imgPaths) || imgPaths.length === 0) {
      return NextResponse.json({ error: 'No image paths provided' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();

    let deleted = 0;
    const failed: string[] = [];

    for (const imgPath of imgPaths) {
      if (typeof imgPath !== 'string') continue;
      if (!imgPath.startsWith(datasetsPath) && !imgPath.startsWith(trainingPath)) {
        failed.push(imgPath);
        continue;
      }
      if (!ALLOWED_EXT.test(imgPath.toLowerCase())) {
        failed.push(imgPath);
        continue;
      }
      try {
        if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);
        const captionPath = imgPath.replace(/\.[^/.]+$/, '') + '.txt';
        if (fs.existsSync(captionPath)) fs.unlinkSync(captionPath);
        deleted++;
      } catch (err) {
        console.error('Failed to delete', imgPath, err);
        failed.push(imgPath);
      }
    }

    return NextResponse.json({ deleted, failed });
  } catch (error) {
    console.error('bulkDelete error', error);
    return NextResponse.json({ error: 'Failed to delete images' }, { status: 500 });
  }
}
