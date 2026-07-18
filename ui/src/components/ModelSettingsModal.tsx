'use client';

import { useState } from 'react';
import { apiClient } from '@/utils/api';
import { Boxes, X, Loader2, Check } from 'lucide-react';
import ModelsTab from '@/app/settings/ModelsTab';
import type { Settings } from '@/hooks/useSettings';

interface Props {
  open: boolean;
  onClose: () => void;
  /** Shared settings from the parent form — editing these updates the form live. */
  settings: Settings;
  setSettings: (updater: (prev: Settings) => Settings) => void;
  /** Fill Name or Path from a scanned installed model, then close. */
  onSelectModel?: (path: string) => void;
}

// A compact, in-place editor for the same options as Settings → Models.
// Because it mutates the parent form's `settings` state, the Model Architecture
// dropdown and Models Folder update on the Training Job form immediately; the
// Save button persists those choices to the backend.
export default function ModelSettingsModal({ open, onClose, settings, setSettings, onSelectModel }: Props) {
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  if (!open) return null;

  const save = () => {
    setStatus('saving');
    apiClient
      .post('/api/settings', settings)
      .then(() => setStatus('success'))
      .catch(err => {
        console.error('Error saving settings:', err);
        setStatus('error');
      })
      .finally(() => {
        setTimeout(() => setStatus('idle'), 2000);
      });
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg shadow-2xl w-full max-w-3xl flex flex-col max-h-[88vh]"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-3 border-b border-gray-800">
          <Boxes className="w-5 h-5 text-blue-400" />
          <div className="flex-1 font-medium">Model Settings</div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body: the same controls as Settings → Models, wired to the form's live settings */}
        <div className="px-5 py-4 overflow-auto">
          <ModelsTab
            modelsFolder={settings.MODELS_FOLDER}
            onModelsFolderChange={value => setSettings(prev => ({ ...prev, MODELS_FOLDER: value }))}
            enabledModelArchs={settings.ENABLED_MODEL_ARCHS}
            onEnabledModelArchsChange={value => setSettings(prev => ({ ...prev, ENABLED_MODEL_ARCHS: value }))}
            onSelectModel={
              onSelectModel
                ? path => {
                    onSelectModel(path);
                    onClose();
                  }
                : undefined
            }
          />
        </div>

        {/* Footer */}
        <div className="flex items-center gap-2 px-5 py-3 border-t border-gray-800">
          <div className="flex-1 text-xs">
            {status === 'success' && (
              <span className="text-green-400 flex items-center gap-1">
                <Check className="w-3.5 h-3.5" /> Saved
              </span>
            )}
            {status === 'error' && <span className="text-red-400">Error saving settings.</span>}
            <span className="text-gray-500">
              {status === 'idle' && 'Changes update the form immediately. Save to keep them.'}
            </span>
          </div>
          <button
            onClick={onClose}
            className="px-3 py-2 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 rounded-lg"
          >
            Close
          </button>
          <button
            onClick={save}
            disabled={status === 'saving'}
            className="px-4 py-2 text-sm text-white bg-blue-700 hover:bg-blue-600 rounded-lg disabled:opacity-50 flex items-center gap-2"
          >
            {status === 'saving' && <Loader2 className="w-4 h-4 animate-spin" />}
            {status === 'saving' ? 'Saving…' : 'Save Settings'}
          </button>
        </div>
      </div>
    </div>
  );
}
