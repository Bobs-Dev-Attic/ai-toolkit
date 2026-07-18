'use client';

import { Fragment, useEffect, useMemo, useState } from 'react';
import { apiClient } from '@/utils/api';
import { modelArchs } from '@/app/jobs/new/options';
import { Folder, Loader2, Search, Check, X, ChevronDown, ChevronRight } from 'lucide-react';

interface InstalledModel {
  path: string;
  name: string;
  rel: string;
  ext: string;
  size: number;
  modified_at: number;
  metadata?: {
    tensor_count?: number;
    param_count?: number;
    dtypes?: string[];
    arch_hints?: string[];
  };
}

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

function formatParams(n: number | undefined): string {
  if (!n) return '—';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} K`;
  return `${n}`;
}

interface Props {
  modelsFolder: string;
  onModelsFolderChange: (value: string) => void;
  enabledModelArchs: string;
  onEnabledModelArchsChange: (value: string) => void;
  /**
   * When provided, each installed model row gets a "Use" action that passes the
   * model's absolute path back (used by the in-form Model Settings modal to fill
   * in Name or Path). Omitted on the standalone Settings page.
   */
  onSelectModel?: (path: string) => void;
}

export default function ModelsTab({
  modelsFolder,
  onModelsFolderChange,
  enabledModelArchs,
  onEnabledModelArchsChange,
  onSelectModel,
}: Props) {
  const [installed, setInstalled] = useState<InstalledModel[]>([]);
  const [scanLoading, setScanLoading] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [scannedFolder, setScannedFolder] = useState<string>('');
  const [search, setSearch] = useState('');
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [archSearch, setArchSearch] = useState('');

  const enabledSet = useMemo(() => {
    try {
      if (enabledModelArchs) {
        const parsed = JSON.parse(enabledModelArchs);
        if (Array.isArray(parsed)) return new Set(parsed.filter((s: any) => typeof s === 'string'));
      }
    } catch {}
    return new Set<string>();
  }, [enabledModelArchs]);
  const allEnabled = enabledSet.size === 0;

  const refreshScan = () => {
    setScanLoading(true);
    setScanError(null);
    apiClient
      .get('/api/models/list')
      .then(res => {
        const data = res.data || {};
        setInstalled(data.models || []);
        setScannedFolder(data.folder || '');
        if (data.error) setScanError(data.error);
      })
      .catch(err => {
        console.error(err);
        setScanError('Failed to scan models folder.');
      })
      .finally(() => setScanLoading(false));
  };

  useEffect(() => {
    refreshScan();
  }, []);

  const groupedArchs = useMemo(() => {
    const g: Record<string, { name: string; label: string }[]> = {};
    for (const a of modelArchs) {
      const group = (a as any).group || 'image';
      (g[group] ||= []).push({ name: a.name, label: a.label });
    }
    return Object.entries(g)
      .map(([group, items]) => ({
        group,
        items: items.sort((a, b) => a.label.localeCompare(b.label)),
      }))
      .sort((a, b) => a.group.localeCompare(b.group));
  }, []);

  const filteredArchGroups = useMemo(() => {
    if (!archSearch.trim()) return groupedArchs;
    const q = archSearch.toLowerCase();
    return groupedArchs
      .map(({ group, items }) => ({
        group,
        items: items.filter(
          a => a.label.toLowerCase().includes(q) || a.name.toLowerCase().includes(q),
        ),
      }))
      .filter(g => g.items.length > 0);
  }, [groupedArchs, archSearch]);

  const setEnabled = (next: Set<string>) => {
    if (next.size === 0) {
      onEnabledModelArchsChange('');
      return;
    }
    onEnabledModelArchsChange(JSON.stringify(Array.from(next).sort()));
  };

  const toggleArch = (name: string) => {
    if (allEnabled) {
      // Initialize from "all" by enabling everything except the one being toggled off.
      const everything = new Set(modelArchs.map(a => a.name));
      everything.delete(name);
      setEnabled(everything);
      return;
    }
    const next = new Set(enabledSet);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setEnabled(next);
  };

  const selectAll = () => onEnabledModelArchsChange('');
  const selectNone = () => onEnabledModelArchsChange(JSON.stringify([]));

  const filteredInstalled = useMemo(() => {
    if (!search.trim()) return installed;
    const q = search.toLowerCase();
    return installed.filter(
      m => m.rel.toLowerCase().includes(q) || m.name.toLowerCase().includes(q),
    );
  }, [installed, search]);

  return (
    <div className="space-y-10">
      {/* ───── Section 1: Models folder ───── */}
      <section>
        <h2 className="text-lg font-medium mb-1 flex items-center gap-2">
          <Folder className="w-5 h-5" /> Models Folder
        </h2>
        <p className="text-sm text-gray-400 mb-3">
          Absolute path on this machine where downloaded model checkpoints live. ai-toolkit will scan
          this folder (up to 5 levels deep) for <code className="bg-gray-800 px-1 rounded">.safetensors</code>,{' '}
          <code className="bg-gray-800 px-1 rounded">.ckpt</code>,{' '}
          <code className="bg-gray-800 px-1 rounded">.bin</code>,{' '}
          <code className="bg-gray-800 px-1 rounded">.pt</code>, and{' '}
          <code className="bg-gray-800 px-1 rounded">.gguf</code> files.
        </p>
        <input
          type="text"
          value={modelsFolder}
          onChange={e => onModelsFolderChange(e.target.value)}
          placeholder="e.g. C:\Users\you\models or /mnt/models"
          className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent text-sm"
          spellCheck={false}
        />
      </section>

      {/* ───── Section 2: Enabled architectures ───── */}
      <section>
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-lg font-medium">Architectures in Job Dropdown</h2>
          <div className="text-xs text-gray-400">
            {allEnabled ? 'All architectures enabled' : `${enabledSet.size} of ${modelArchs.length} enabled`}
          </div>
        </div>
        <p className="text-sm text-gray-400 mb-3">
          Pick which model architectures show up in the Model Architecture dropdown on the New / Edit
          Training Job page. Leave everything checked (or clear all then check the ones you use) to
          hide families you never train against.
        </p>
        <div className="flex items-center gap-2 mb-3">
          <div className="relative flex-1 max-w-sm">
            <Search className="w-4 h-4 absolute left-2 top-2 text-gray-500" />
            <input
              type="text"
              value={archSearch}
              onChange={e => setArchSearch(e.target.value)}
              placeholder="Filter architectures..."
              className="w-full bg-gray-800 border border-gray-700 rounded pl-8 pr-3 py-1.5 text-sm"
            />
          </div>
          <button
            type="button"
            onClick={selectAll}
            className="text-xs text-blue-400 hover:text-blue-300 px-2 py-1 rounded hover:bg-gray-800"
          >
            All
          </button>
          <button
            type="button"
            onClick={selectNone}
            className="text-xs text-blue-400 hover:text-blue-300 px-2 py-1 rounded hover:bg-gray-800"
          >
            None
          </button>
        </div>
        <div className="space-y-3 border border-gray-800 rounded-lg p-3 max-h-96 overflow-auto">
          {filteredArchGroups.length === 0 ? (
            <div className="text-gray-500 text-sm">No architectures match your filter.</div>
          ) : (
            filteredArchGroups.map(({ group, items }) => (
              <div key={group}>
                <div className="text-[10px] uppercase tracking-wide text-gray-500 mb-1">{group}</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-x-3 gap-y-1">
                  {items.map(arch => {
                    const enabled = allEnabled || enabledSet.has(arch.name);
                    return (
                      <label
                        key={arch.name}
                        className="flex items-center gap-2 text-sm cursor-pointer hover:bg-gray-800/60 rounded px-1 py-0.5"
                      >
                        <input
                          type="checkbox"
                          checked={enabled}
                          onChange={() => toggleArch(arch.name)}
                        />
                        <span className={enabled ? 'text-gray-100' : 'text-gray-500'}>
                          {arch.label}
                        </span>
                        <span className="text-[10px] text-gray-500 font-mono">{arch.name}</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      {/* ───── Section 3: Installed models scan ───── */}
      <section>
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-lg font-medium">Installed Models</h2>
          <button
            type="button"
            onClick={refreshScan}
            className="text-xs text-blue-400 hover:text-blue-300 px-2 py-1 rounded hover:bg-gray-800"
          >
            Refresh scan
          </button>
        </div>
        <p className="text-sm text-gray-400 mb-3">
          {scannedFolder ? (
            <>
              Scanning <code className="bg-gray-800 px-1 rounded">{scannedFolder}</code> — found{' '}
              <span className="text-gray-200">{installed.length}</span> file
              {installed.length === 1 ? '' : 's'}.
            </>
          ) : (
            'Configure a Models Folder above, then save settings, to scan installed checkpoints.'
          )}
        </p>

        {scanError && (
          <div className="text-xs text-amber-300 bg-amber-900/20 border border-amber-700/40 rounded p-2 mb-3">
            <X className="w-3 h-3 inline mr-1" /> {scanError}
          </div>
        )}

        {scannedFolder && (
          <>
            <div className="relative max-w-sm mb-2">
              <Search className="w-4 h-4 absolute left-2 top-2 text-gray-500" />
              <input
                type="text"
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Filter installed models..."
                className="w-full bg-gray-800 border border-gray-700 rounded pl-8 pr-3 py-1.5 text-sm"
              />
            </div>
            <div className="border border-gray-800 rounded-lg overflow-auto max-h-[28rem]">
              {scanLoading ? (
                <div className="p-6 flex items-center justify-center text-gray-400 text-sm">
                  <Loader2 className="w-4 h-4 animate-spin mr-2" /> Scanning…
                </div>
              ) : filteredInstalled.length === 0 ? (
                <div className="p-6 text-center text-gray-500 text-sm">No model files found.</div>
              ) : (
                <table className="w-full text-sm">
                  <thead className="bg-gray-800 text-gray-400 text-xs uppercase">
                    <tr>
                      <th className="px-3 py-2 text-left w-6"></th>
                      <th className="px-3 py-2 text-left">Path</th>
                      <th className="px-3 py-2 text-right w-24">Size</th>
                      <th className="px-3 py-2 text-right w-24">Params</th>
                      <th className="px-3 py-2 text-left w-32">Dtype</th>
                      <th className="px-3 py-2 text-right w-36">Modified</th>
                      {onSelectModel && <th className="px-3 py-2 text-right w-20"></th>}
                    </tr>
                  </thead>
                  <tbody>
                    {filteredInstalled.map(m => {
                      const isOpen = expandedRow === m.path;
                      const meta = m.metadata;
                      return (
                        <Fragment key={m.path}>
                          <tr
                            className="border-t border-gray-800 hover:bg-gray-900/60 cursor-pointer"
                            onClick={() => setExpandedRow(isOpen ? null : m.path)}
                          >
                            <td className="px-2 py-1.5 text-gray-500">
                              {isOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                            </td>
                            <td className="px-3 py-1.5 font-mono text-xs text-gray-200 truncate" title={m.path}>
                              {m.rel || m.name}
                            </td>
                            <td className="px-3 py-1.5 text-right tabular-nums text-gray-300">
                              {formatBytes(m.size)}
                            </td>
                            <td className="px-3 py-1.5 text-right tabular-nums text-gray-300">
                              {formatParams(meta?.param_count)}
                            </td>
                            <td className="px-3 py-1.5 text-gray-300">
                              {meta?.dtypes?.join(', ') || '—'}
                            </td>
                            <td className="px-3 py-1.5 text-right text-gray-400 whitespace-nowrap">
                              {new Date(m.modified_at).toLocaleString()}
                            </td>
                            {onSelectModel && (
                              <td className="px-3 py-1.5 text-right">
                                <button
                                  type="button"
                                  onClick={e => {
                                    e.stopPropagation();
                                    onSelectModel(m.path);
                                  }}
                                  className="inline-flex items-center gap-1 text-xs text-blue-400 hover:text-white bg-blue-900/30 hover:bg-blue-700 px-2 py-1 rounded"
                                >
                                  <Check className="w-3 h-3" /> Use
                                </button>
                              </td>
                            )}
                          </tr>
                          {isOpen && (
                            <tr className="bg-gray-950/60">
                              <td></td>
                              <td colSpan={onSelectModel ? 6 : 5} className="px-3 py-2 text-xs text-gray-300 space-y-1">
                                <div>
                                  <span className="text-gray-500">Full path:</span>{' '}
                                  <span className="font-mono">{m.path}</span>
                                </div>
                                <div>
                                  <span className="text-gray-500">Extension:</span>{' '}
                                  <span className="font-mono">{m.ext}</span>
                                </div>
                                {meta?.tensor_count !== undefined && (
                                  <div>
                                    <span className="text-gray-500">Tensor count:</span>{' '}
                                    <span className="font-mono">{meta.tensor_count.toLocaleString()}</span>
                                  </div>
                                )}
                                {meta?.arch_hints && meta.arch_hints.length > 0 && (
                                  <div>
                                    <span className="text-gray-500">Detected components:</span>{' '}
                                    {meta.arch_hints.map(h => (
                                      <span
                                        key={h}
                                        className="inline-block text-[11px] bg-blue-900/40 text-blue-200 px-2 py-0.5 rounded-full mr-1"
                                      >
                                        {h}
                                      </span>
                                    ))}
                                  </div>
                                )}
                                {!meta && (
                                  <div className="text-gray-500 italic">
                                    No tensor metadata — file isn't a .safetensors checkpoint.
                                  </div>
                                )}
                              </td>
                            </tr>
                          )}
                        </Fragment>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </>
        )}
      </section>
    </div>
  );
}
