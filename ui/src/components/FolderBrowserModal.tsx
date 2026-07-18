'use client';

import { useEffect, useState, useCallback } from 'react';
import { apiClient } from '@/utils/api';
import { Folder, FileBox, File as FileIcon, ArrowUp, X, Loader2, HardDrive } from 'lucide-react';

interface DirEntry {
  name: string;
  path: string;
}
interface FileEntry extends DirEntry {
  size: number;
  isModel: boolean;
}

interface BrowseResponse {
  path: string;
  parent: string | null;
  sep: string;
  dirs: DirEntry[];
  files: FileEntry[];
}

interface Props {
  open: boolean;
  onClose: () => void;
  /** Where to start browsing. Falls back to the configured Models Folder. */
  initialPath?: string;
  /** Called with the chosen folder or model file path. */
  onSelect: (selectedPath: string) => void;
}

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

export default function FolderBrowserModal({ open, onClose, initialPath, onSelect }: Props) {
  const [data, setData] = useState<BrowseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Editable path box so the user can also type/paste a location to jump to.
  const [pathInput, setPathInput] = useState('');

  const browse = useCallback((target?: string) => {
    setLoading(true);
    setError(null);
    apiClient
      .get('/api/browse', { params: target ? { path: target } : {} })
      .then(res => {
        const d = res.data as BrowseResponse;
        setData(d);
        setPathInput(d.path);
      })
      .catch(err => {
        console.error('Browse error:', err);
        setError(err?.response?.data?.error || 'Unable to read that location.');
      })
      .finally(() => setLoading(false));
  }, []);

  // (Re)load whenever the modal is opened.
  useEffect(() => {
    if (open) browse(initialPath || undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  // The path bar is editable, so a typed/pasted location that was never loaded
  // (no Enter, no "Go") must still be what we select — resolve it first, and
  // keep the modal open on a bad path rather than silently selecting the folder
  // the user happens to be viewing.
  const chooseFolder = async () => {
    const typed = pathInput.trim();
    if (typed && data?.path && typed !== data.path) {
      setLoading(true);
      setError(null);
      try {
        const res = await apiClient.get('/api/browse', { params: { path: typed } });
        const d = res.data as BrowseResponse;
        setData(d);
        setPathInput(d.path);
        onSelect(d.path);
        onClose();
      } catch (err: any) {
        console.error('Browse error:', err);
        setError(err?.response?.data?.error || 'Unable to read that location.');
      } finally {
        setLoading(false);
      }
      return;
    }
    if (data?.path) {
      onSelect(data.path);
      onClose();
    }
  };

  const chooseFile = (filePath: string) => {
    onSelect(filePath);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg shadow-2xl w-full max-w-2xl flex flex-col max-h-[85vh]"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-3 border-b border-gray-800">
          <Folder className="w-5 h-5 text-blue-400" />
          <div className="flex-1 font-medium">Browse for model folder or file</div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Path bar */}
        <div className="flex items-center gap-2 px-5 py-3 border-b border-gray-800">
          <button
            type="button"
            onClick={() => data?.parent && browse(data.parent)}
            disabled={!data?.parent || loading}
            title="Up one level"
            className="p-1.5 rounded bg-gray-800 hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <ArrowUp className="w-4 h-4" />
          </button>
          <input
            type="text"
            value={pathInput}
            onChange={e => setPathInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') browse(pathInput.trim());
            }}
            spellCheck={false}
            className="flex-1 min-w-0 bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-sm font-mono text-gray-200"
          />
          <button
            type="button"
            onClick={() => browse(pathInput.trim())}
            className="px-3 py-1.5 text-sm rounded bg-gray-800 hover:bg-gray-700"
          >
            Go
          </button>
        </div>

        {/* Listing */}
        <div className="flex-1 overflow-auto min-h-[16rem]">
          {loading ? (
            <div className="p-8 flex items-center justify-center text-gray-400 text-sm">
              <Loader2 className="w-4 h-4 animate-spin mr-2" /> Loading…
            </div>
          ) : error ? (
            <div className="p-8 text-center text-amber-300 text-sm">{error}</div>
          ) : data ? (
            <div className="py-1">
              {data.dirs.length === 0 && data.files.length === 0 && (
                <div className="p-8 text-center text-gray-500 text-sm">This folder is empty.</div>
              )}
              {data.dirs.map(d => (
                <button
                  key={d.path}
                  type="button"
                  onDoubleClick={() => browse(d.path)}
                  onClick={() => browse(d.path)}
                  className="w-full flex items-center gap-2 px-5 py-1.5 text-sm text-left hover:bg-gray-800/70"
                >
                  <Folder className="w-4 h-4 text-blue-400 shrink-0" />
                  <span className="truncate">{d.name}</span>
                </button>
              ))}
              {data.files.map(f => (
                <button
                  key={f.path}
                  type="button"
                  onClick={() => f.isModel && chooseFile(f.path)}
                  disabled={!f.isModel}
                  title={f.isModel ? 'Select this model file' : 'Not a model file'}
                  className={`w-full flex items-center gap-2 px-5 py-1.5 text-sm text-left ${
                    f.isModel ? 'hover:bg-gray-800/70 cursor-pointer' : 'opacity-40 cursor-default'
                  }`}
                >
                  {f.isModel ? (
                    <FileBox className="w-4 h-4 text-emerald-400 shrink-0" />
                  ) : (
                    <FileIcon className="w-4 h-4 text-gray-500 shrink-0" />
                  )}
                  <span className="truncate flex-1">{f.name}</span>
                  <span className="text-xs text-gray-500 tabular-nums shrink-0">{formatBytes(f.size)}</span>
                </button>
              ))}
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-2 px-5 py-3 border-t border-gray-800">
          <div className="flex items-center gap-1.5 text-xs text-gray-500 flex-1 min-w-0">
            <HardDrive className="w-3.5 h-3.5 shrink-0" />
            <span className="truncate">Pick a model file, or choose the current folder.</span>
          </div>
          <button
            onClick={onClose}
            className="px-3 py-2 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 rounded-lg"
          >
            Cancel
          </button>
          <button
            onClick={chooseFolder}
            disabled={!data?.path}
            className="px-4 py-2 text-sm text-white bg-blue-700 hover:bg-blue-600 rounded-lg disabled:opacity-40"
          >
            Select This Folder
          </button>
        </div>
      </div>
    </div>
  );
}
