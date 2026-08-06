import path from 'path';
import prisma from './prisma';

export const TOOLKIT_ROOT = path.resolve('@', '..', '..');
export const defaultTrainFolder = path.join(TOOLKIT_ROOT, 'output');
export const defaultDatasetsFolder = path.join(TOOLKIT_ROOT, 'datasets');
export const defaultDataRoot = path.join(TOOLKIT_ROOT, 'data');
export const defaultModelsFolder = path.join(TOOLKIT_ROOT, 'models');

// Forked file-server workers set AI_TOOLKIT_QUIET_PATHS so this line prints
// once per launched process group, not once per worker.
if (!process.env.AI_TOOLKIT_QUIET_PATHS) {
  console.log('TOOLKIT_ROOT:', TOOLKIT_ROOT);
}

export const getTrainingFolder = async () => {
  const key = 'TRAINING_FOLDER';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let trainingRoot = defaultTrainFolder;
  if (row?.value && row.value !== '') {
    trainingRoot = row.value;
  }
  return trainingRoot as string;
};

export const getHFToken = async () => {
  const key = 'HF_TOKEN';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let token = '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  return token;
};

// User-configured HuggingFace hub cache location (empty = default). Injected as
// HF_HUB_CACHE into the training subprocess so downloads land on the chosen drive.
export const getHfHubCache = async () => {
  let row = await prisma.settings.findFirst({
    where: {
      key: 'HF_HUB_CACHE',
    },
  });
  if (row?.value && row.value.trim() !== '') {
    return row.value.trim();
  }
  return '';
};

export const getModelsPath = async () => {
  const key = 'MODELS_PATH';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let modelsPath = '';
  if (row?.value && row.value !== '' && row.value !== defaultModelsFolder) {
    modelsPath = row.value;
  }
  return modelsPath;
};
