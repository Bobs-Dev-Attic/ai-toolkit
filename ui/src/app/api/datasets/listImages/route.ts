import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { imageSize } from 'image-size';
import { getDatasetsRoot } from '@/server/settings';

const STATIC_IMAGE_EXT = new Set(['.png', '.jpg', '.jpeg', '.webp']);

export async function POST(request: Request) {
  const datasetsPath = await getDatasetsRoot();
  const body = await request.json();
  const { datasetName } = body;
  const datasetFolder = path.join(datasetsPath, datasetName);

  try {
    // Check if folder exists
    if (!fs.existsSync(datasetFolder)) {
      return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
    }

    // Find all images recursively
    const imageFiles = findImagesRecursively(datasetFolder);

    // Sort server-side so the client doesn't have to sort large lists.
    imageFiles.sort((a, b) => a.localeCompare(b));

    const result = imageFiles.map(imgPath => {
      const item: { img_path: string; size?: number; width?: number; height?: number } = { img_path: imgPath };
      try {
        item.size = fs.statSync(imgPath).size;
        // Reading the image header is cheap (a few KB), much faster than
        // decoding pixels. Only attempt it for still-image formats.
        if (STATIC_IMAGE_EXT.has(path.extname(imgPath).toLowerCase())) {
          const dims = imageSize(fs.readFileSync(imgPath));
          item.width = dims.width;
          item.height = dims.height;
        }
      } catch {
        // best-effort; leave the fields undefined
      }
      return item;
    });

    return NextResponse.json({ images: result });
  } catch (error) {
    console.error('Error finding images:', error);
    return NextResponse.json({ error: 'Failed to process request' }, { status: 500 });
  }
}

/**
 * Recursively finds all image files in a directory and its subdirectories
 * @param dir Directory to search
 * @returns Array of absolute paths to image files
 */
function findImagesRecursively(dir: string): string[] {
  const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.mp3', '.wav', '.flac', '.ogg'];
  let results: string[] = [];

  // withFileTypes avoids a separate statSync per entry — a big win on large datasets
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const name = entry.name;
    if (name.startsWith('.')) continue;
    const itemPath = path.join(dir, name);

    if (entry.isDirectory()) {
      if (name === '_controls') continue;
      results = results.concat(findImagesRecursively(itemPath));
    } else if (entry.isFile()) {
      const ext = path.extname(name).toLowerCase();
      if (imageExtensions.includes(ext)) {
        results.push(itemPath);
      }
    }
  }

  return results;
}
