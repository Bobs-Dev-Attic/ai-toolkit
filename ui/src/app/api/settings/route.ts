import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { defaultTrainFolder, defaultDatasetsFolder } from '@/paths';
import { flushCache } from '@/server/settings';

const prisma = new PrismaClient();

export async function GET() {
  try {
    const settings = await prisma.settings.findMany();
    const settingsObject = settings.reduce((acc: any, setting) => {
      acc[setting.key] = setting.value;
      return acc;
    }, {});
    if (!settingsObject.TRAINING_FOLDER || settingsObject.TRAINING_FOLDER === '') {
      settingsObject.TRAINING_FOLDER = defaultTrainFolder;
    }
    if (!settingsObject.DATASETS_FOLDER || settingsObject.DATASETS_FOLDER === '') {
      settingsObject.DATASETS_FOLDER = defaultDatasetsFolder;
    }
    if (!settingsObject.MODELS_FOLDER) {
      settingsObject.MODELS_FOLDER = '';
    }
    if (!settingsObject.ENABLED_MODEL_ARCHS) {
      settingsObject.ENABLED_MODEL_ARCHS = '';
    }
    if (!settingsObject.HF_HUB_CACHE) {
      settingsObject.HF_HUB_CACHE = '';
    }
    return NextResponse.json(settingsObject);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch settings' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { HF_TOKEN, TRAINING_FOLDER, DATASETS_FOLDER, MODELS_FOLDER, ENABLED_MODEL_ARCHS, HF_HUB_CACHE } = body;

    const ops: Promise<any>[] = [];
    const upsert = (key: string, value: string | undefined | null) => {
      if (value === undefined) return;
      const v = value ?? '';
      ops.push(
        prisma.settings.upsert({
          where: { key },
          update: { value: v },
          create: { key, value: v },
        }),
      );
    };
    upsert('HF_TOKEN', HF_TOKEN);
    upsert('TRAINING_FOLDER', TRAINING_FOLDER);
    upsert('DATASETS_FOLDER', DATASETS_FOLDER);
    upsert('MODELS_FOLDER', MODELS_FOLDER);
    upsert('ENABLED_MODEL_ARCHS', ENABLED_MODEL_ARCHS);
    upsert('HF_HUB_CACHE', HF_HUB_CACHE);

    await Promise.all(ops);
    flushCache();
    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
