'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface Settings {
  HF_TOKEN: string;
  TRAINING_FOLDER: string;
  DATASETS_FOLDER: string;
  MODELS_FOLDER: string;
  /** JSON-encoded array of modelArch names. Empty string = all enabled. */
  ENABLED_MODEL_ARCHS: string;
  /** Override for the HuggingFace hub cache dir. Empty = default location. */
  HF_HUB_CACHE: string;
  MODELS_PATH: string;
}

export default function useSettings() {
  const [settings, setSettings] = useState<Settings>({
    HF_TOKEN: '',
    TRAINING_FOLDER: '',
    DATASETS_FOLDER: '',
    MODELS_FOLDER: '',
    ENABLED_MODEL_ARCHS: '',
    HF_HUB_CACHE: '',
    MODELS_PATH: '',
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    apiClient
      .get('/api/settings')
      .then(res => res.data)
      .then(data => {
        console.log('Settings:', data);
        setSettings({
          HF_TOKEN: data.HF_TOKEN || '',
          TRAINING_FOLDER: data.TRAINING_FOLDER || '',
          DATASETS_FOLDER: data.DATASETS_FOLDER || '',
          MODELS_FOLDER: data.MODELS_FOLDER || '',
          ENABLED_MODEL_ARCHS: data.ENABLED_MODEL_ARCHS || '',
          HF_HUB_CACHE: data.HF_HUB_CACHE || '',
          MODELS_PATH: data.MODELS_PATH || '',
        });
        setIsLoaded(true);
      })
      .catch(error => console.error('Error fetching settings:', error));
  }, []);

  return { settings, setSettings, isSettingsLoaded };
}
