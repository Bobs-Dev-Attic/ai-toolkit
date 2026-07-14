'use client';

import { useState, useEffect } from 'react';
import useSettings from '@/hooks/useSettings';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import ModelsTab from './ModelsTab';
import RestartModal from '@/components/RestartModal';
import FolderBrowserModal from '@/components/FolderBrowserModal';
import { Settings as SettingsIcon, Boxes, Server, FolderOpen, RotateCcw, TriangleAlert, CircleCheck, Loader2 } from 'lucide-react';

type Tab = 'general' | 'models' | 'server';

const GB = 1024 ** 3;

function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes < 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

interface HfCacheCurrent {
  path: string;
  isDefault: boolean;
  sizeBytes: number;
  freeBytes: number | null;
  totalBytes: number | null;
}
interface HfCacheCandidate {
  path: string;
  exists: boolean;
  writable: boolean;
  freeBytes: number | null;
  totalBytes: number | null;
}

export default function Settings() {
  const { settings, setSettings } = useSettings();
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [tab, setTab] = useState<Tab>('general');
  const [restartOpen, setRestartOpen] = useState(false);

  // HuggingFace cache location state
  const [hfCache, setHfCache] = useState<HfCacheCurrent | null>(null);
  const [hfBrowseOpen, setHfBrowseOpen] = useState(false);
  const [hfCandidate, setHfCandidate] = useState<HfCacheCandidate | null>(null);
  const [hfChecking, setHfChecking] = useState(false);

  useEffect(() => {
    apiClient
      .get('/api/hf-cache')
      .then(r => setHfCache(r.data.current))
      .catch(() => {});
  }, []);

  // Pick a folder -> ask the server to format it and report the target drive.
  const onPickHfFolder = (folder: string) => {
    setHfChecking(true);
    apiClient
      .get('/api/hf-cache', { params: { candidate: folder } })
      .then(r => {
        const cand = r.data.candidate as HfCacheCandidate;
        if (cand) {
          setHfCandidate(cand);
          setSettings(prev => ({ ...prev, HF_HUB_CACHE: cand.path }));
        }
      })
      .catch(() => {})
      .finally(() => setHfChecking(false));
  };

  const resetHfCache = () => {
    setHfCandidate(null);
    setSettings(prev => ({ ...prev, HF_HUB_CACHE: '' }));
  };

  // ── Move existing cache ─────────────────────────────────────────────
  const [moveStatus, setMoveStatus] = useState<{
    state: 'running' | 'done' | 'error';
    sourceBytes: number;
    destBytes?: number;
    error?: string;
  } | null>(null);
  const [moveStarting, setMoveStarting] = useState(false);

  const startMove = () => {
    if (!hfCache || !hfCandidate) return;
    const sizeStr = formatBytes(hfCache.sizeBytes);
    if (
      !window.confirm(
        `Move ${sizeStr} of cached models from\n\n${hfCache.path}\n\nto\n\n${hfCandidate.path}\n\n` +
          `This can take a while on a different drive, and you should not train while it runs. Continue?`,
      )
    ) {
      return;
    }
    setMoveStarting(true);
    apiClient
      .post('/api/hf-cache/move', { source: hfCache.path, dest: hfCandidate.path })
      .then(r => {
        setMoveStatus(r.data);
      })
      .catch(err => {
        setMoveStatus({ state: 'error', sourceBytes: 0, error: err?.response?.data?.error || 'Failed to start move.' });
      })
      .finally(() => setMoveStarting(false));
  };

  // Poll move progress while running.
  useEffect(() => {
    if (moveStatus?.state !== 'running') return;
    const t = setInterval(() => {
      apiClient
        .get('/api/hf-cache/move')
        .then(r => {
          setMoveStatus(r.data);
          if (r.data.state === 'done') {
            // refresh the current-cache panel and clear the candidate
            apiClient.get('/api/hf-cache').then(res => setHfCache(res.data.current)).catch(() => {});
            setHfCandidate(null);
          }
        })
        .catch(() => {});
    }, 2000);
    return () => clearInterval(t);
  }, [moveStatus?.state]);

  const movePct =
    moveStatus && moveStatus.sourceBytes > 0
      ? Math.min(100, ((moveStatus.destBytes ?? 0) / moveStatus.sourceBytes) * 100)
      : 0;

  // Adequacy of the chosen target drive (only evaluated once a candidate is picked).
  const currentCacheSize = hfCache?.sizeBytes ?? 0;
  let hfTier: 'ok' | 'warn' | 'block' | null = null;
  if (hfCandidate) {
    const free = hfCandidate.freeBytes ?? 0;
    if (!hfCandidate.writable || free < 20 * GB) hfTier = 'block';
    else if (free < currentCacheSize) hfTier = 'warn';
    else hfTier = 'ok';
  }
  const hfCacheBlocked = hfTier === 'block';

  const currentPathDisplay = settings.HF_HUB_CACHE || hfCache?.path || 'Loading…';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('saving');
    apiClient
      .post('/api/settings', settings)
      .then(() => setStatus('success'))
      .catch(error => {
        console.error('Error saving settings:', error);
        setStatus('error');
      })
      .finally(() => {
        setTimeout(() => setStatus('idle'), 2000);
      });
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSettings(prev => ({ ...prev, [name]: value }));
  };

  const tabs: { id: Tab; label: string; icon: React.ComponentType<any> }[] = [
    { id: 'general', label: 'General', icon: SettingsIcon },
    { id: 'models', label: 'Models', icon: Boxes },
    { id: 'server', label: 'Server', icon: Server },
  ];

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Settings</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <div className="flex items-center gap-1 border-b border-gray-800 mb-6">
          {tabs.map(t => {
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={`flex items-center gap-2 px-4 py-2 text-sm border-b-2 -mb-px transition-colors ${
                  active
                    ? 'border-blue-500 text-white'
                    : 'border-transparent text-gray-400 hover:text-gray-200'
                }`}
              >
                <t.icon className="w-4 h-4" />
                {t.label}
              </button>
            );
          })}
        </div>

        {tab !== 'server' && (
          <form onSubmit={handleSubmit} className="space-y-6">
            {tab === 'general' && (
              <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                <div>
                  <div className="space-y-4">
                    <div>
                      <label htmlFor="HF_TOKEN" className="block text-sm font-medium mb-2">
                        Hugging Face Token
                        <div className="text-gray-500 text-sm ml-1">
                          Create a Read token on{' '}
                          <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                            {' '}Huggingface
                          </a>{' '}
                          if you need to access gated/private models.
                        </div>
                      </label>
                      <input
                        type="password"
                        id="HF_TOKEN"
                        name="HF_TOKEN"
                        value={settings.HF_TOKEN}
                        onChange={handleChange}
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                        placeholder="Enter your Hugging Face token"
                      />
                    </div>

                    {/* HuggingFace cache location */}
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Hugging Face Cache Location
                        <div className="text-gray-500 text-sm ml-1 font-normal">
                          Where downloaded models are stored. Most people don't want this on their main drive — pick a
                          folder on a larger disk.
                        </div>
                      </label>

                      {/* Current path + size */}
                      <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5">
                        <div className="flex items-center gap-2 text-xs text-gray-400">
                          <span>Current</span>
                          {hfCache?.isDefault && !settings.HF_HUB_CACHE && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-300">default</span>
                          )}
                          {hfCache && (
                            <span className="ml-auto tabular-nums">
                              {formatBytes(hfCache.sizeBytes)} used
                              {hfCache.freeBytes != null && <> · {formatBytes(hfCache.freeBytes)} free on drive</>}
                            </span>
                          )}
                        </div>
                        <div className="font-mono text-sm text-gray-100 break-all mt-1">{currentPathDisplay}</div>
                        <div className="flex items-center gap-3 mt-2">
                          <button
                            type="button"
                            onClick={() => setHfBrowseOpen(true)}
                            className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded bg-gray-700 hover:bg-gray-600"
                          >
                            <FolderOpen className="w-3.5 h-3.5" /> Browse…
                          </button>
                          {(settings.HF_HUB_CACHE || hfCandidate) && (
                            <button
                              type="button"
                              onClick={resetHfCache}
                              className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded text-gray-300 hover:bg-gray-800"
                            >
                              <RotateCcw className="w-3.5 h-3.5" /> Reset to default
                            </button>
                          )}
                          {hfChecking && (
                            <span className="inline-flex items-center gap-1 text-xs text-gray-400">
                              <Loader2 className="w-3.5 h-3.5 animate-spin" /> checking space…
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Candidate check result */}
                      {hfCandidate && hfTier && (
                        <div
                          className={`mt-2 rounded-lg border px-3 py-2 text-sm ${
                            hfTier === 'block'
                              ? 'border-rose-500/40 bg-rose-500/5 text-rose-300'
                              : hfTier === 'warn'
                                ? 'border-amber-500/40 bg-amber-500/5 text-amber-300'
                                : 'border-emerald-500/40 bg-emerald-500/5 text-emerald-300'
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            {hfTier === 'ok' ? <CircleCheck className="w-4 h-4" /> : <TriangleAlert className="w-4 h-4" />}
                            <span className="font-mono text-xs break-all text-gray-200">{hfCandidate.path}</span>
                          </div>
                          <div className="mt-1 text-xs">
                            {hfCandidate.freeBytes != null && (
                              <>
                                {formatBytes(hfCandidate.freeBytes)} free of {formatBytes(hfCandidate.totalBytes)} ·{' '}
                              </>
                            )}
                            {hfTier === 'block' && !hfCandidate.writable && 'This folder is not writable. '}
                            {hfTier === 'block' && hfCandidate.writable && 'Not enough free space to be usable. '}
                            {hfTier === 'block' && "You can't save until you pick a location with more room."}
                            {hfTier === 'warn' &&
                              `Less free space than your current cache (${formatBytes(currentCacheSize)}). Fine for new downloads, but the existing cache won't fully fit if you move it.`}
                            {hfTier === 'ok' && 'Adequate space. New downloads will go here.'}
                          </div>

                          {/* Move existing cache */}
                          {hfTier !== 'block' && (!moveStatus || moveStatus.state === 'error') && (
                            <div className="mt-2 flex items-center gap-3">
                              <button
                                type="button"
                                onClick={startMove}
                                disabled={moveStarting}
                                className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded bg-blue-700 hover:bg-blue-600 text-white disabled:opacity-50"
                              >
                                {moveStarting ? (
                                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                ) : (
                                  <FolderOpen className="w-3.5 h-3.5" />
                                )}
                                Move existing cache here ({formatBytes(hfCache?.sizeBytes)})
                              </button>
                              {moveStatus?.state === 'error' && (
                                <span className="text-xs text-rose-400">{moveStatus.error}</span>
                              )}
                            </div>
                          )}

                          {moveStatus?.state === 'running' && (
                            <div className="mt-2">
                              <div className="flex items-center justify-between text-xs text-gray-300 mb-1">
                                <span>Moving cache…</span>
                                <span className="tabular-nums">
                                  {formatBytes(moveStatus.destBytes)} / {formatBytes(moveStatus.sourceBytes)} (
                                  {movePct.toFixed(0)}%)
                                </span>
                              </div>
                              <div className="w-full bg-gray-800 rounded-full h-2">
                                <div
                                  className="h-2 rounded-full bg-blue-500 transition-all"
                                  style={{ width: `${movePct}%` }}
                                />
                              </div>
                              <div className="text-[11px] text-gray-500 mt-1">
                                Runs in the background — you can leave this page. Don't start training until it finishes.
                              </div>
                            </div>
                          )}

                          {moveStatus?.state === 'done' && (
                            <div className="mt-2 text-xs text-emerald-400 flex items-center gap-1.5">
                              <CircleCheck className="w-3.5 h-3.5" /> Cache moved successfully.
                            </div>
                          )}
                        </div>
                      )}

                      <p className="text-gray-500 text-xs mt-2">
                        Changing this only affects <span className="text-gray-400">future</span> downloads. Your existing
                        cache isn't moved automatically — copy it to the new location if you want to reuse it, or it will
                        re-download on next use.
                      </p>
                    </div>

                    <div>
                      <label htmlFor="TRAINING_FOLDER" className="block text-sm font-medium mb-2">
                        Training Folder Path
                        <div className="text-gray-500 text-sm ml-1">
                          We will store your training information here. Must be an absolute path. If blank, it will
                          default to the output folder in the project root.
                        </div>
                      </label>
                      <input
                        type="text"
                        id="TRAINING_FOLDER"
                        name="TRAINING_FOLDER"
                        value={settings.TRAINING_FOLDER}
                        onChange={handleChange}
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                        placeholder="Enter training folder path"
                      />
                    </div>

                    <div>
                      <label htmlFor="DATASETS_FOLDER" className="block text-sm font-medium mb-2">
                        Dataset Folder Path
                        <div className="text-gray-500 text-sm ml-1">
                          Where we store and find your datasets.{' '}
                          <span className="text-orange-800">
                            Warning: This software may modify datasets so it is recommended you keep a backup somewhere
                            else or have a dedicated folder for this software.
                          </span>
                        </div>
                      </label>
                      <input
                        type="text"
                        id="DATASETS_FOLDER"
                        name="DATASETS_FOLDER"
                        value={settings.DATASETS_FOLDER}
                        onChange={handleChange}
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                        placeholder="Enter datasets folder path"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {tab === 'models' && (
              <ModelsTab
                modelsFolder={settings.MODELS_FOLDER}
                onModelsFolderChange={value => setSettings(prev => ({ ...prev, MODELS_FOLDER: value }))}
                enabledModelArchs={settings.ENABLED_MODEL_ARCHS}
                onEnabledModelArchsChange={value =>
                  setSettings(prev => ({ ...prev, ENABLED_MODEL_ARCHS: value }))
                }
              />
            )}

            <button
              type="submit"
              disabled={status === 'saving' || hfCacheBlocked}
              className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {status === 'saving' ? 'Saving...' : 'Save Settings'}
            </button>

            {hfCacheBlocked && (
              <p className="text-rose-400 text-center text-sm">
                Pick a Hugging Face cache location with adequate free space before saving.
              </p>
            )}
            {status === 'success' && <p className="text-green-500 text-center">Settings saved successfully!</p>}
            {status === 'error' && <p className="text-red-500 text-center">Error saving settings. Please try again.</p>}
          </form>
        )}

        {tab === 'server' && (
          <div className="max-w-2xl space-y-6">
            <div>
              <h2 className="text-lg font-medium mb-1">Stop server</h2>
              <p className="text-sm text-gray-400 mb-3">
                Gracefully exits the Next.js process. The command window hosting{' '}
                <code className="bg-gray-800 px-1 rounded">Start-AI-Toolkit.bat</code> will close on its own once the
                .bat script finishes. To bring the server back, double-click{' '}
                <code className="bg-gray-800 px-1 rounded">Start-AI-Toolkit.bat</code> again — it will detect and kill
                any orphan listener on port 8675 before starting fresh.
              </p>
              <button
                type="button"
                onClick={() => setRestartOpen(true)}
                className="px-4 py-2 rounded-lg bg-red-700 hover:bg-red-600 text-white"
              >
                Stop Server…
              </button>
            </div>
          </div>
        )}
      </MainContent>

      <RestartModal open={restartOpen} onClose={() => setRestartOpen(false)} />
      <FolderBrowserModal
        open={hfBrowseOpen}
        onClose={() => setHfBrowseOpen(false)}
        initialPath={settings.HF_HUB_CACHE || hfCache?.path || undefined}
        onSelect={onPickHfFolder}
      />
    </>
  );
}
