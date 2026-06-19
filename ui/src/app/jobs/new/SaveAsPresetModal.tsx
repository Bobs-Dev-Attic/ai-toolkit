'use client';

import { useEffect, useMemo, useState } from 'react';
import { JobConfig } from '@/types';
import { apiClient } from '@/utils/api';
import { Save, X, Sparkles } from 'lucide-react';
import classNames from 'classnames';

interface Props {
  open: boolean;
  onClose: () => void;
  jobConfig: JobConfig;
  archName: string;
  archLabel?: string;
  onSaved?: () => void;
}

interface FieldDef {
  path: string;
  label: string;
  group: 'Model' | 'Training' | 'Dataset' | 'Save / Sample';
}

// Curated list of settings that make sense in a preset. Mirrors what the
// built-in CONFIG_PRESETS capture so user presets feel symmetric with them.
const FIELDS: FieldDef[] = [
  { group: 'Model', path: 'config.process[0].model.quantize', label: 'Quantize transformer' },
  { group: 'Model', path: 'config.process[0].model.quantize_te', label: 'Quantize text encoder' },
  { group: 'Model', path: 'config.process[0].model.low_vram', label: 'Low VRAM mode' },
  { group: 'Model', path: 'config.process[0].model.layer_offloading', label: 'Layer offloading' },
  {
    group: 'Model',
    path: 'config.process[0].model.layer_offloading_transformer_percent',
    label: 'Layer offloading: transformer %',
  },
  {
    group: 'Model',
    path: 'config.process[0].model.layer_offloading_text_encoder_percent',
    label: 'Layer offloading: text encoder %',
  },
  { group: 'Training', path: 'config.process[0].train.batch_size', label: 'Batch size' },
  { group: 'Training', path: 'config.process[0].train.gradient_accumulation', label: 'Gradient accumulation' },
  { group: 'Training', path: 'config.process[0].train.gradient_checkpointing', label: 'Gradient checkpointing' },
  { group: 'Training', path: 'config.process[0].train.lr', label: 'Learning rate' },
  { group: 'Training', path: 'config.process[0].train.steps', label: 'Total steps' },
  { group: 'Training', path: 'config.process[0].train.optimizer', label: 'Optimizer' },
  {
    group: 'Dataset',
    path: 'config.process[0].datasets[0].cache_latents_to_disk',
    label: 'Cache latents to disk',
  },
  { group: 'Dataset', path: 'config.process[0].datasets[0].resolution', label: 'Resolution' },
  { group: 'Save / Sample', path: 'config.process[0].save.save_every', label: 'Save every (steps)' },
  { group: 'Save / Sample', path: 'config.process[0].sample.sample_every', label: 'Sample every (steps)' },
];

// Resolve a dotted path with array indices on a config object.
function getAtPath(obj: any, path: string): any {
  if (obj == null) return undefined;
  const segments = path
    .replace(/\[(\d+)\]/g, '.$1')
    .split('.')
    .filter(Boolean);
  let cur = obj;
  for (const seg of segments) {
    if (cur == null) return undefined;
    cur = cur[seg];
  }
  return cur;
}

function fmtValue(v: any): string {
  if (v === undefined) return '—';
  if (v === null) return 'null';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (Array.isArray(v)) return `[${v.join(', ')}]`;
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

export default function SaveAsPresetModal({ open, onClose, jobConfig, archName, archLabel, onSaved }: Props) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [vram, setVram] = useState('');
  const [includeAllArchs, setIncludeAllArchs] = useState(false);
  const [saving, setSaving] = useState(false);
  const [included, setIncluded] = useState<Record<string, boolean>>({});

  // When opened, pre-select every field that has a concrete value in the
  // current config — users would expect a snapshot to capture what's there.
  useEffect(() => {
    if (!open) return;
    const next: Record<string, boolean> = {};
    for (const f of FIELDS) {
      const v = getAtPath(jobConfig, f.path);
      next[f.path] = v !== undefined && v !== null;
    }
    setIncluded(next);
    setName('');
    setDescription('');
    setVram('');
    setIncludeAllArchs(false);
  }, [open, jobConfig]);

  const grouped = useMemo(() => {
    const g: Record<string, FieldDef[]> = {};
    for (const f of FIELDS) {
      (g[f.group] ||= []).push(f);
    }
    return g;
  }, []);

  const selectedCount = Object.values(included).filter(Boolean).length;

  if (!open) return null;

  const toggle = (path: string) => setIncluded(s => ({ ...s, [path]: !s[path] }));

  const save = async () => {
    if (!name.trim()) {
      alert('Give the preset a name.');
      return;
    }
    const overrides: Record<string, any> = {};
    for (const f of FIELDS) {
      if (!included[f.path]) continue;
      const v = getAtPath(jobConfig, f.path);
      if (v === undefined) continue;
      overrides[f.path] = v;
    }
    if (Object.keys(overrides).length === 0) {
      alert('Pick at least one field to include in the preset.');
      return;
    }

    setSaving(true);
    try {
      const existing = await apiClient.get('/api/presets').then(r => r.data?.presets || []);
      const newPreset = {
        id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        name: name.trim(),
        description: description.trim(),
        modelArchs: includeAllArchs ? [] : archName ? [archName] : [],
        overrides,
        approxVramGB: vram.trim() === '' ? undefined : Number(vram),
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      await apiClient.post('/api/presets', { presets: [newPreset, ...existing] });
      onSaved?.();
      onClose();
    } catch (err: any) {
      alert(err?.response?.data?.error || 'Failed to save preset.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg w-[95vw] max-w-2xl max-h-[90vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center px-4 py-3 border-b border-gray-800">
          <Sparkles className="w-5 h-5 text-amber-400 mr-2" />
          <div className="text-lg flex-1">Save Configuration as Preset</div>
          <button className="text-gray-300 hover:text-white" onClick={onClose}>
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-auto p-4 space-y-3 text-sm">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <div className="text-xs text-gray-400 mb-1">Preset name</div>
              <input
                type="text"
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder="e.g. My 24 GB Balanced"
                className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100"
              />
            </div>
            <div>
              <div className="text-xs text-gray-400 mb-1">Approx VRAM (GB, optional)</div>
              <input
                type="text"
                inputMode="numeric"
                value={vram}
                onChange={e => setVram(e.target.value.replace(/[^0-9.]/g, ''))}
                placeholder="e.g. 20"
                className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100"
              />
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1">Description</div>
            <input
              type="text"
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="One-line description shown on the card"
              className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100"
            />
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs text-gray-300">
            <span>Apply to:</span>
            {archName && (
              <label className="flex items-center gap-1">
                <input
                  type="radio"
                  name="archScope"
                  checked={!includeAllArchs}
                  onChange={() => setIncludeAllArchs(false)}
                />
                Just <span className="text-blue-300">{archLabel || archName}</span>
              </label>
            )}
            <label className="flex items-center gap-1">
              <input
                type="radio"
                name="archScope"
                checked={includeAllArchs}
                onChange={() => setIncludeAllArchs(true)}
              />
              All model architectures (assign later on /presets)
            </label>
          </div>

          <div>
            <div className="text-xs text-gray-400 mb-1 flex items-center justify-between">
              <span>Fields to include ({selectedCount} of {FIELDS.length})</span>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => {
                    const allOn: Record<string, boolean> = {};
                    FIELDS.forEach(f => (allOn[f.path] = true));
                    setIncluded(allOn);
                  }}
                  className="text-blue-400 hover:underline"
                >
                  Select all
                </button>
                <button
                  type="button"
                  onClick={() => setIncluded({})}
                  className="text-blue-400 hover:underline"
                >
                  Clear
                </button>
              </div>
            </div>
            <div className="border border-gray-800 rounded bg-gray-950 p-2 space-y-3 max-h-64 overflow-auto">
              {Object.entries(grouped).map(([group, items]) => (
                <div key={group}>
                  <div className="text-[10px] uppercase tracking-wide text-gray-500 mb-1">{group}</div>
                  <div className="space-y-1">
                    {items.map(f => {
                      const v = getAtPath(jobConfig, f.path);
                      const present = v !== undefined && v !== null;
                      const sel = !!included[f.path];
                      return (
                        <label
                          key={f.path}
                          className={classNames(
                            'flex items-center gap-2 text-xs cursor-pointer rounded px-1 py-0.5',
                            sel ? 'text-gray-100' : 'text-gray-400',
                            !present && 'opacity-60',
                          )}
                        >
                          <input type="checkbox" checked={sel} onChange={() => toggle(f.path)} />
                          <span className="flex-1 truncate">{f.label}</span>
                          <span className="font-mono text-gray-500 text-[11px] truncate max-w-[40%]">
                            {fmtValue(v)}
                          </span>
                        </label>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
            <div className="text-[11px] text-gray-500 mt-1">
              Each checked field gets stored as a dotted-path override (
              <code className="bg-gray-800 px-1 rounded">config.process[0].train.batch_size</code>) — the same format
              built-in presets use. Edit later on the Preset Configurations page.
            </div>
          </div>
        </div>

        <div className="px-4 py-3 border-t border-gray-800 flex items-center justify-end gap-2">
          <button
            onClick={onClose}
            className="px-3 py-1 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 rounded"
          >
            Cancel
          </button>
          <button
            onClick={save}
            disabled={saving || selectedCount === 0 || !name.trim()}
            className="px-3 py-1 text-sm text-white bg-green-600 hover:bg-green-700 rounded flex items-center gap-1 disabled:opacity-40"
          >
            <Save className="w-3 h-3" /> {saving ? 'Saving…' : 'Save Preset'}
          </button>
        </div>
      </div>
    </div>
  );
}
