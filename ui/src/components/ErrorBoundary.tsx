// ErrorBoundary.tsx
'use client';

import React, { ReactNode, ErrorInfo, Component } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  /**
   * Fallback UI when a child throws. Pass a function to receive the actual
   * error so the fallback can render its message — otherwise pass a static
   * ReactNode.
   */
  fallback?: ReactNode | ((error: Error | null) => ReactNode);
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('Error caught by ErrorBoundary:', error, errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      const fallback = this.props.fallback;
      if (typeof fallback === 'function') {
        return fallback(this.state.error);
      }
      return fallback || <div>Something went wrong.</div>;
    }
    return this.props.children;
  }
}

export default ErrorBoundary;