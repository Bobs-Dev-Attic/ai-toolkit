'use client';

import { useEffect, useRef, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { defaultJobConfig, defaultDatasetConfig, migrateJobConfig } from './jobConfig';
import { jobTypeOptions, modelArchs } from './options';
import { JobConfig } from '@/types';
import { objectCopy } from '@/utils/basic';
import { useNestedState, setNestedValue } from '@/utils/hooks';
import { SelectInput } from '@/components/formInputs';
import useSettings from '@/hooks/useSettings';
import useGPUInfo from '@/hooks/useGPUInfo';
import useDatasetList from '@/hooks/useDatasetList';
import YAML from 'yaml';
import path from 'path';
import { TopBar, MainContent } from '@/components/layout';
import { Button, Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { ChevronDown, Save } from 'lucide-react';
import SaveAsPresetModal from './SaveAsPresetModal';
import PreflightModal from '@/components/PreflightModal';
import { FaChevronLeft } from 'react-icons/fa';
import SimpleJob from './SimpleJob';
import AdvancedConfigEditor from '@/components/AdvancedConfigEditor';
import ErrorBoundary from '@/components/ErrorBoundary';
import { apiClient } from '@/utils/api';
import SplitWorkspace, { ChangeEntry } from './SplitWorkspace';

const isDev = process.env.NODE_ENV === 'development';

export default function TrainingForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const runId = searchParams.get('id');
  const cloneId = searchParams.get('cloneId');
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const { settings, isSettingsLoaded } = useSettings();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const { datasets, status: datasetFetchStatus } = useDatasetList();
  const [datasetOptions, setDatasetOptions] = useState<{ value: string; label: string }[]>([]);
  const [viewMode, setViewMode] = useState<'simple' | 'advanced' | 'split'>('simple');
  const showAdvancedView = viewMode === 'advanced';
  const isSplit = viewMode === 'split';
  const [savePresetOpen, setSavePresetOpen] = useState(false);

  // Track recent edits in split mode for the cross-pane highlight overlay.
  const [splitChanges, setSplitChanges] = useState<ChangeEntry[]>([]);
  const changeIdRef = useRef(0);
  const recordChange = (origin: 'simple' | 'advanced', value: any, path?: string) => {
    if (!isSplit || !path) return;
    const id = ++changeIdRef.current;
    setSplitChanges(prev => {
      // Keep at most 64 entries; older ones drop off when the 30s window expires too.
      const next = [...prev, { id, path, value, ts: Date.now(), origin }];
      return next.length > 64 ? next.slice(next.length - 64) : next;
    });
  };
  const setJobConfigFromSimple = (value: any, path?: string) => {
    recordChange('simple', value, path);
    setJobConfig(value, path);
  };
  const setJobConfigFromAdvanced = (value: any, path?: string) => {
    recordChange('advanced', value, path);
    setJobConfig(value, path);
  };
  // Drop stale changes after 30s to keep the list bounded.
  useEffect(() => {
    if (!isSplit) return;
    const t = setInterval(() => {
      const cutoff = Date.now() - 31_000;
      setSplitChanges(prev => prev.filter(c => c.ts > cutoff));
    }, 5000);
    return () => clearInterval(t);
  }, [isSplit]);

  const [jobConfig, setJobConfig] = useNestedState<JobConfig>(objectCopy(migrateJobConfig(defaultJobConfig)));
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImportConfig = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = reader.result as string;
        let parsed: any;
        if (file.name.endsWith('.json') || file.name.endsWith('.jsonc')) {
          parsed = JSON.parse(text.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, ''));
        } else {
          parsed = YAML.parse(text);
        }

        // Set required fields (same pattern as AdvancedJob.handleChange)
        try {
          parsed.config.process[0].sqlite_db_path = './aitk_db.db';
          parsed.config.process[0].training_folder = settings.TRAINING_FOLDER;
          parsed.config.process[0].device = 'cuda';
          parsed.config.process[0].performance_log_every = 10;
        } catch (err) {
          console.warn('Could not set required fields on imported config:', err);
        }

        migrateJobConfig(parsed);
        setJobConfig(parsed);
      } catch (err) {
        console.error('Failed to parse config file:', err);
        alert('Failed to parse config file. Please check the file format.');
      }
    };
    reader.readAsText(file);

    // Reset so the same file can be re-imported
    e.target.value = '';
  };

  useEffect(() => {
    if (!isSettingsLoaded) return;
    if (datasetFetchStatus !== 'success') return;

    // Base options immediately; enrich labels with image counts once stats load.
    const baseOptions = datasets.map(name => ({ value: path.join(settings.DATASETS_FOLDER, name), label: name }));
    setDatasetOptions(baseOptions);

    // Fetch image counts and append them to the dropdown labels, e.g. "melissa (25)".
    // Best-effort: if stats fail, the base labels stay.
    apiClient
      .get('/api/datasets/stats')
      .then(r => {
        const counts = new Map<string, number>((r.data?.datasets ?? []).map((d: any) => [d.name, d.image_count]));
        setDatasetOptions(
          datasets.map(name => {
            const c = counts.get(name);
            return {
              value: path.join(settings.DATASETS_FOLDER, name),
              label: c != null ? `${name} (${c})` : name,
            };
          }),
        );
      })
      .catch(() => {
        /* keep base labels */
      });

    if (baseOptions.length > 0) {
      const defaultDatasetPath = defaultDatasetConfig.folder_path;
      // Use functional updater so we check the *current* state, not a stale closure
      setJobConfig((prev: JobConfig) => {
        let updated = prev;
        for (let i = 0; i < prev.config.process[0].datasets.length; i++) {
          if (prev.config.process[0].datasets[i].folder_path === defaultDatasetPath) {
            updated = setNestedValue(updated, baseOptions[0].value, `config.process[0].datasets[${i}].folder_path`);
          }
        }
        return updated;
      });
    }
  }, [datasets, settings, isSettingsLoaded, datasetFetchStatus]);

  // clone existing job
  useEffect(() => {
    if (cloneId) {
      apiClient
        .get(`/api/jobs?id=${cloneId}`)
        .then(res => res.data)
        .then(data => {
          console.log('Clone Training:', data);
          setGpuIDs(data.gpu_ids);
          const newJobConfig = migrateJobConfig(JSON.parse(data.job_config));
          newJobConfig.config.name = `${newJobConfig.config.name}_copy`;
          setJobConfig(newJobConfig);
        })
        .catch(error => console.error('Error fetching training:', error));
    }
  }, [cloneId]);

  useEffect(() => {
    if (runId) {
      apiClient
        .get(`/api/jobs?id=${runId}`)
        .then(res => res.data)
        .then(data => {
          console.log('Training:', data);
          setGpuIDs(data.gpu_ids);
          setJobConfig(migrateJobConfig(JSON.parse(data.job_config)));
        })
        .catch(error => console.error('Error fetching training:', error));
    }
  }, [runId]);

  useEffect(() => {
    if (isGPUInfoLoaded) {
      if (gpuIDs === null && gpuList.length > 0) {
        setGpuIDs(`${gpuList[0].index}`);
      }
    }
  }, [gpuList, isGPUInfoLoaded]);

  useEffect(() => {
    if (isSettingsLoaded) {
      setJobConfig(settings.TRAINING_FOLDER, 'config.process[0].training_folder');
    }
  }, [settings, isSettingsLoaded]);

  const saveJob = async (asDraft = false) => {
    if (status === 'saving') return;
    setStatus('saving');

    apiClient
      .post('/api/jobs', {
        id: runId,
        name: jobConfig.config.name,
        gpu_ids: gpuIDs,
        job_config: jobConfig,
        ...(asDraft ? { status: 'draft' } : {}),
      })
      .then(res => {
        setStatus('success');
        if (asDraft) {
          router.push('/jobs/drafts');
          return;
        }
        if (runId) {
          router.push(`/jobs/${runId}`);
        } else {
          router.push(`/jobs/${res.data.id}`);
        }
      })
      .catch(error => {
        if (error.response?.status === 409) {
          alert('Training name already exists. Please choose a different name.');
        } else {
          alert('Failed to save job. Please try again.');
        }
        console.log('Error saving training:', error);
      })
      .finally(() =>
        setTimeout(() => {
          setStatus('idle');
        }, 2000),
      );
  };

  const [preflightOpen, setPreflightOpen] = useState(false);

  // Whether this config is a trainable job worth pre-flighting (has model + train).
  const isTrainingJob = !!jobConfig?.config?.process?.[0]?.train && !!jobConfig?.config?.process?.[0]?.model;

  // Create flow: for training jobs, show the pre-flight check first; otherwise save directly.
  const requestCreate = () => {
    if (status === 'saving') return;
    if (isTrainingJob) {
      setPreflightOpen(true);
    } else {
      saveJob(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    requestCreate();
  };

  return (
    <>
      <TopBar>
        <div className="flex-shrink-0">
          <Button className="text-gray-500 dark:text-gray-300 px-2 sm:px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div className="flex-shrink-0">
          <h1 className="text-base sm:text-lg truncate max-w-[120px] sm:max-w-none">
            {runId ? 'Edit Training Job' : 'New Training Job'}
          </h1>
        </div>
        <div className="flex-1"></div>
        {(showAdvancedView || isSplit) && (
          <>
            <div className="hidden sm:block">
              <SelectInput
                value={`${gpuIDs}`}
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            </div>
            <div className="hidden sm:block mx-4 bg-gray-200 dark:bg-gray-800 w-1 h-6"></div>
            <div className="hidden md:block">
              <Button className="text-gray-200 bg-gray-800 px-3 py-1 rounded-md" onClick={handleImportConfig}>
                Import Config
              </Button>
            </div>
            <div className="hidden md:block mx-4 bg-gray-200 dark:bg-gray-800 w-1 h-6"></div>
          </>
        )}
        {!showAdvancedView && (
          <>
            <div className="hidden sm:block">
              <SelectInput
                value={`${jobConfig?.config.process[0].type}`}
                onChange={value => {
                  // undo current job type changes
                  const currentOption = jobTypeOptions.find(
                    option => option.value === jobConfig?.config.process[0].type,
                  );
                  if (currentOption && currentOption.onDeactivate) {
                    setJobConfig(currentOption.onDeactivate(objectCopy(jobConfig)));
                  }
                  const option = jobTypeOptions.find(option => option.value === value);
                  if (option) {
                    if (option.onActivate) {
                      setJobConfig(option.onActivate(objectCopy(jobConfig)));
                    }
                    jobTypeOptions.forEach(opt => {
                      if (opt.value !== option.value && opt.onDeactivate) {
                        setJobConfig(opt.onDeactivate(objectCopy(jobConfig)));
                      }
                    });
                  }
                  setJobConfig(value, 'config.process[0].type');
                }}
                options={jobTypeOptions}
              />
            </div>
            <div className="hidden sm:block mx-4 bg-gray-200 dark:bg-gray-800 w-1 h-6"></div>
          </>
        )}

        <div className="pr-1 sm:pr-2 flex bg-gray-800 rounded-md overflow-hidden text-xs sm:text-sm flex-shrink-0">
          {(['simple', 'advanced', 'split'] as const).map(mode => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              className={`px-2 sm:px-3 py-1 capitalize ${
                viewMode === mode ? 'bg-slate-600 text-white' : 'text-gray-300 hover:bg-gray-700'
              }`}
              title={mode === 'split' ? 'Simple + Advanced side-by-side (live-syncs)' : undefined}
            >
              {mode}
            </button>
          ))}
        </div>
        <div className="pr-1 sm:pr-2">
          <Menu>
            <MenuButton
              disabled={status === 'saving'}
              className="flex items-center gap-1 text-white bg-amber-600 hover:bg-amber-700 disabled:opacity-50 px-2 sm:px-3 py-1 rounded-md text-xs sm:text-base"
            >
              <Save className="w-3.5 h-3.5" />
              <span className="sm:hidden">Save</span>
              <span className="hidden sm:inline">Save As…</span>
              <ChevronDown className="w-3.5 h-3.5" />
            </MenuButton>
            <MenuItems
              anchor={{ to: 'bottom end', gap: 6 }}
              className="bg-gray-900 border border-gray-700 rounded shadow-lg w-56 py-1 z-50 text-sm focus:outline-none"
            >
              <MenuItem>
                {({ focus }) => (
                  <button
                    type="button"
                    onClick={() => saveJob(true)}
                    className={`w-full flex items-start gap-2 px-3 py-2 text-left ${
                      focus ? 'bg-gray-800' : ''
                    }`}
                  >
                    <div className="text-amber-400 mt-0.5">📝</div>
                    <div className="flex-1">
                      <div className="text-gray-100">Save as Draft</div>
                      <div className="text-xs text-gray-400">
                        Park this config under Draft Jobs without queuing it.
                      </div>
                    </div>
                  </button>
                )}
              </MenuItem>
              <MenuItem>
                {({ focus }) => (
                  <button
                    type="button"
                    onClick={() => setSavePresetOpen(true)}
                    className={`w-full flex items-start gap-2 px-3 py-2 text-left ${
                      focus ? 'bg-gray-800' : ''
                    }`}
                  >
                    <div className="text-purple-400 mt-0.5">✨</div>
                    <div className="flex-1">
                      <div className="text-gray-100">Save as Preset</div>
                      <div className="text-xs text-gray-400">
                        Capture current settings as a reusable preset for any future job.
                      </div>
                    </div>
                  </button>
                )}
              </MenuItem>
            </MenuItems>
          </Menu>
        </div>
        <div className="flex-shrink-0">
          <Button
            className="text-white bg-green-600 hover:bg-green-700 px-2 sm:px-3 py-1 rounded-md text-xs sm:text-base"
            onClick={requestCreate}
            disabled={status === 'saving'}
          >
            {status === 'saving' ? (
              'Saving...'
            ) : (
              <>
                <span className="sm:hidden">{runId ? 'Update' : 'Create'}</span>
                <span className="hidden sm:inline">{runId ? 'Update Job' : 'Create Job'}</span>
              </>
            )}
          </Button>
        </div>
      </TopBar>

      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml,.json,.jsonc"
        style={{ display: 'none' }}
        onChange={handleFileSelected}
      />

      {isSplit ? (
        <SplitWorkspace
          leftLabel="Simple"
          rightLabel="Advanced — edits sync live"
          changes={splitChanges}
          leftPane={
            <div className="p-4">
              <ErrorBoundary
                fallback={
                  <div className="flex items-center justify-center h-64 text-sm text-red-600 font-medium bg-red-100 dark:bg-red-900/20 dark:text-red-400 border border-red-300 dark:border-red-700 rounded-lg">
                    Advanced-only job detected. Use the Advanced pane on the right.
                  </div>
                }
              >
                <SimpleJob
                  jobConfig={jobConfig}
                  setJobConfig={setJobConfigFromSimple}
                  status={status}
                  handleSubmit={handleSubmit}
                  runId={runId}
                  gpuIDs={gpuIDs}
                  setGpuIDs={setGpuIDs}
                  gpuList={gpuList}
                  datasetOptions={datasetOptions}
                  isLoading={!isSettingsLoaded || !isGPUInfoLoaded || datasetFetchStatus !== 'success'}
                />
              </ErrorBoundary>
              <div className="pt-12"></div>
            </div>
          }
          rightPane={
            <AdvancedConfigEditor
              config={jobConfig}
              setConfig={setJobConfigFromAdvanced}
              transformOnParse={(parsed: any) => {
                try {
                  parsed.config.process[0].sqlite_db_path = './aitk_db.db';
                  parsed.config.process[0].training_folder = settings.TRAINING_FOLDER;
                  parsed.config.process[0].device = 'cuda';
                  parsed.config.process[0].performance_log_every = 10;
                } catch (e) {
                  console.warn(e);
                }
                return migrateJobConfig(parsed);
              }}
            />
          }
        />
      ) : showAdvancedView ? (
        <div className="pt-[48px] absolute top-0 left-0 w-full h-full overflow-auto">
          <AdvancedConfigEditor
            config={jobConfig}
            setConfig={setJobConfig}
            transformOnParse={(parsed: any) => {
              try {
                parsed.config.process[0].sqlite_db_path = './aitk_db.db';
                parsed.config.process[0].training_folder = settings.TRAINING_FOLDER;
                parsed.config.process[0].device = 'cuda';
                parsed.config.process[0].performance_log_every = 10;
              } catch (e) {
                console.warn(e);
              }
              return migrateJobConfig(parsed);
            }}
          />
        </div>
      ) : (
        <MainContent>
          <ErrorBoundary
            fallback={err => (
              <div className="bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-700 rounded-lg p-4 my-4">
                <div className="font-medium mb-1">The Simple form couldn't render this job.</div>
                <div className="text-sm mb-2">
                  Either the config has fields the Simple form doesn't support, or there's a bug.
                  Switch to <span className="font-semibold">Advanced</span> view to inspect / edit the raw config.
                </div>
                {err?.message && (
                  <details className="text-xs mt-2">
                    <summary className="cursor-pointer">Show error details</summary>
                    <pre className="mt-2 p-2 bg-black/30 rounded overflow-auto max-h-40 whitespace-pre-wrap">
                      {err.message}
                      {err.stack ? `\n\n${err.stack}` : ''}
                    </pre>
                  </details>
                )}
              </div>
            )}
          >
            <SimpleJob
              jobConfig={jobConfig}
              setJobConfig={setJobConfig}
              status={status}
              handleSubmit={handleSubmit}
              runId={runId}
              gpuIDs={gpuIDs}
              setGpuIDs={setGpuIDs}
              gpuList={gpuList}
              datasetOptions={datasetOptions}
              isLoading={!isSettingsLoaded || !isGPUInfoLoaded || datasetFetchStatus !== 'success'}
            />
          </ErrorBoundary>

          <div className="pt-20"></div>
        </MainContent>
      )}
      <SaveAsPresetModal
        open={savePresetOpen}
        onClose={() => setSavePresetOpen(false)}
        jobConfig={jobConfig}
        archName={jobConfig.config.process[0].model.arch}
        archLabel={
          modelArchs.find(a => a.name === jobConfig.config.process[0].model.arch)?.label
        }
      />
      <PreflightModal
        open={preflightOpen}
        jobConfig={jobConfig}
        onConfirm={() => {
          setPreflightOpen(false);
          saveJob(false);
        }}
        onCancel={() => setPreflightOpen(false)}
        onApplyFixes={fixes => {
          // Apply each suggestion's path/value into the job config in one update.
          setJobConfig((prev: JobConfig) => {
            let updated = prev;
            for (const fix of fixes) {
              updated = setNestedValue(updated, fix.value, fix.path);
            }
            return updated;
          });
        }}
      />
    </>
  );
}
