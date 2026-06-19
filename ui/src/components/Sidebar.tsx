'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Home, Settings, BrainCircuit, Images, Plus, X, FileText, BookText, Sliders } from 'lucide-react';
import { FaXTwitter, FaDiscord, FaYoutube } from 'react-icons/fa6';
import { createGlobalState } from 'react-global-hooks';
import classNames from 'classnames';
import ThemeToggle from './ThemeToggle';
import ThemeLogo from './ThemeLogo';
import ActiveJobWidget from './ActiveJobWidget';
import OstrisCloudBalance from './OstrisCloudBalance';

export const mobileSidebarState = createGlobalState<boolean>(false);

const COLLAPSED_STORAGE_KEY = 'AITK_SIDEBAR_COLLAPSED';

// Double-chevron icon (from chevron-left-double-svgrepo). Rotates 180° to
// face right when the sidebar is collapsed.
const DoubleChevron = ({ pointRight, className }: { pointRight?: boolean; className?: string }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    style={pointRight ? { transform: 'scaleX(-1)' } : undefined}
  >
    <path
      d="M18 17L13 12L18 7M11 17L6 12L11 7"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const Sidebar = () => {
  const [isMobileOpen, setIsMobileOpen] = mobileSidebarState.use();
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Close mobile menu on route change
  useEffect(() => {
    setIsMobileOpen(false);
  }, [pathname]);

  // Lock body scroll when mobile menu open
  useEffect(() => {
    if (isMobileOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isMobileOpen]);

  // Restore collapsed state on mount (avoids SSR/CSR mismatch).
  useEffect(() => {
    setMounted(true);
    try {
      setCollapsed(localStorage.getItem(COLLAPSED_STORAGE_KEY) === '1');
    } catch {}
  }, []);

  const toggleCollapsed = () => {
    setCollapsed(c => {
      const next = !c;
      try {
        localStorage.setItem(COLLAPSED_STORAGE_KEY, next ? '1' : '0');
      } catch {}
      return next;
    });
  };

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home },
    { name: 'New Job', href: '/jobs/new', icon: Plus },
    { name: 'Queue', href: '/jobs', icon: BrainCircuit },
    { name: 'Draft Jobs', href: '/jobs/drafts', icon: FileText },
    { name: 'Prompt Builder', href: '/prompts', icon: BookText },
    { name: 'Preset Configurations', href: '/presets', icon: Sliders },
    { name: 'Datasets', href: '/datasets', icon: Images },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  const socialsBoxClass =
    'flex flex-col items-center justify-center p-1 hover:bg-gray-800 rounded-lg transition-colors';
  const socialIconClass = 'w-5 h-5 text-gray-400 hover:text-white';

  // Don't apply the collapsed style until after mount to keep SSR markup stable.
  const isCollapsed = mounted && collapsed;

  // Mobile drawer always uses the expanded layout — narrow icon-only rail
  // doesn't make sense in a 64px-wide modal.
  const mobileSidebarContent = (
    <>
      <div className="px-4 py-3 flex items-center justify-between">
        <h1 className="text-l">
          <ThemeLogo />
          <span className="font-bold uppercase">Ostris</span>
          <span className="ml-2 uppercase text-gray-300">AI-Toolkit</span>
        </h1>
        <button
          onClick={() => setIsMobileOpen(false)}
          className="md:hidden text-gray-400 hover:text-white p-1"
          aria-label="Close menu"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
      <OstrisCloudBalance />
      <nav className="flex-1">
        <ul className="px-2 py-4 space-y-2">
          {navigation.map(item => (
            <li key={item.name}>
              <Link
                href={item.href}
                className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors"
              >
                <item.icon className="w-5 h-5 mr-3" />
                {item.name}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
      <ActiveJobWidget />
      <a
        href="https://ostris.com/support"
        target="_blank"
        rel="noreferrer"
        className="group flex items-center space-x-2 px-4 py-3 text-gray-400 hover:text-gray-200 transition-colors"
      >
        <svg
          height="20"
          width="20"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          style={{ overflow: 'visible' }}
        >
          <path
            className="animate-heartbeat"
            d="m7 3c-1.5355 0-3.0784 0.5-4.25 1.7-2.3431 2.4-2.2788 6.1 0 8.5l9.25 9.8 9.25-9.8c2.279-2.4 2.343-6.1 0-8.5-2.343-2.3-6.157-2.3-8.5 0l-0.75 0.8-0.75-0.8c-1.172-1.2-2.7145-1.7-4.25-1.7z"
            fill="#c0392b"
          />
        </svg>
        <span className="uppercase text-sm font-medium tracking-wide">Support AI-Toolkit</span>
      </a>

      <div className="px-1 py-1 border-t border-gray-800">
        <div className="grid grid-cols-4 gap-4">
          <a href="https://discord.gg/VXmU2f5WEU" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaDiscord className={socialIconClass} />
          </a>
          <a href="https://www.youtube.com/@ostrisai" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaYoutube className={socialIconClass} />
          </a>
          <a href="https://x.com/ostrisai" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaXTwitter className={socialIconClass} />
          </a>
          <ThemeToggle />
        </div>
      </div>
      <div className="text-center text-[10px] text-gray-400 py-1 bg-gray-800">
        Ostris AI-Toolkit v{process.env.NEXT_PUBLIC_APP_VERSION}
      </div>
    </>
  );

  // Desktop sidebar — collapsible. When collapsed, hides labels and the extra
  // widget sections, leaving just nav icons and the toggle button.
  const desktopSidebar = (
    <div
      className={classNames(
        'hidden md:flex flex-col bg-gray-900 text-gray-100 transition-[width] duration-200',
        isCollapsed ? 'w-14' : 'w-59',
      )}
    >
      <div
        className={classNames(
          'flex items-center py-3',
          isCollapsed ? 'flex-col gap-2 px-1' : 'justify-between px-4',
        )}
      >
        {!isCollapsed && (
          <h1 className="text-l flex items-center">
            <ThemeLogo />
            <span className="font-bold uppercase ml-1">Ostris</span>
            <span className="ml-2 uppercase text-gray-300">AI-Toolkit</span>
          </h1>
        )}
        <button
          onClick={toggleCollapsed}
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="text-gray-400 hover:text-white hover:bg-gray-800 rounded p-1"
        >
          <DoubleChevron pointRight={isCollapsed} className="w-5 h-5" />
        </button>
      </div>

      {!isCollapsed && <OstrisCloudBalance />}

      <nav className="flex-1">
        <ul className={classNames('py-2 space-y-1', isCollapsed ? 'px-1' : 'px-2 space-y-2 py-4')}>
          {navigation.map(item => {
            const active = pathname?.startsWith(item.href);
            return (
              <li key={item.name}>
                <Link
                  href={item.href}
                  title={isCollapsed ? item.name : undefined}
                  className={classNames(
                    'flex items-center text-gray-300 hover:bg-gray-800 rounded-lg transition-colors',
                    isCollapsed ? 'justify-center px-2 py-2' : 'px-4 py-2',
                    active && 'bg-gray-800 text-white',
                  )}
                >
                  <item.icon className={classNames('w-5 h-5', !isCollapsed && 'mr-3')} />
                  {!isCollapsed && <span className="truncate">{item.name}</span>}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {!isCollapsed && (
        <>
          <ActiveJobWidget />
          <a
            href="https://ostris.com/support"
            target="_blank"
            rel="noreferrer"
            className="group flex items-center space-x-2 px-4 py-3 text-gray-400 hover:text-gray-200 transition-colors"
          >
            <svg
              height="20"
              width="20"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              style={{ overflow: 'visible' }}
            >
              <path
                className="animate-heartbeat"
                d="m7 3c-1.5355 0-3.0784 0.5-4.25 1.7-2.3431 2.4-2.2788 6.1 0 8.5l9.25 9.8 9.25-9.8c2.279-2.4 2.343-6.1 0-8.5-2.343-2.3-6.157-2.3-8.5 0l-0.75 0.8-0.75-0.8c-1.172-1.2-2.7145-1.7-4.25-1.7z"
                fill="#c0392b"
              />
            </svg>
            <span className="uppercase text-sm font-medium tracking-wide">Support AI-Toolkit</span>
          </a>

          <div className="px-1 py-1 border-t border-gray-800">
            <div className="grid grid-cols-4 gap-4">
              <a href="https://discord.gg/VXmU2f5WEU" target="_blank" rel="noreferrer" className={socialsBoxClass}>
                <FaDiscord className={socialIconClass} />
              </a>
              <a href="https://www.youtube.com/@ostrisai" target="_blank" rel="noreferrer" className={socialsBoxClass}>
                <FaYoutube className={socialIconClass} />
              </a>
              <a href="https://x.com/ostrisai" target="_blank" rel="noreferrer" className={socialsBoxClass}>
                <FaXTwitter className={socialIconClass} />
              </a>
              <ThemeToggle />
            </div>
          </div>
          <div className="text-center text-[10px] text-gray-400 py-1 bg-gray-800">
            Ostris AI-Toolkit v{process.env.NEXT_PUBLIC_APP_VERSION}
          </div>
        </>
      )}

      {isCollapsed && (
        <div className="border-t border-gray-800 py-2 flex justify-center">
          <ThemeToggle />
        </div>
      )}
    </div>
  );

  return (
    <>
      {desktopSidebar}

      {/* Mobile overlay sidebar */}
      <div
        className={`md:hidden fixed inset-0 bg-black/60 z-40 transition-opacity duration-300 ease-in-out ${
          isMobileOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
        onClick={() => setIsMobileOpen(false)}
        aria-hidden="true"
      />
      <div
        className={`md:hidden fixed top-0 left-0 bottom-0 w-64 max-w-[85vw] bg-gray-900 text-gray-100 z-50 flex flex-col shadow-xl transform transition-transform duration-300 ease-in-out ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {mobileSidebarContent}
      </div>
    </>
  );
};

export default Sidebar;
