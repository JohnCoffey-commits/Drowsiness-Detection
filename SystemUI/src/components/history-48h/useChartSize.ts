"use client";

import { useCallback, useEffect, useState } from "react";

interface ChartSize {
  width: number;
  height: number;
}

export function useChartSize<T extends HTMLElement>() {
  const [node, setNode] = useState<T | null>(null);
  const [size, setSize] = useState<ChartSize>({ width: 0, height: 0 });
  const containerRef = useCallback((nextNode: T | null) => {
    setNode(nextNode);
  }, []);

  useEffect(() => {
    if (!node) return;

    let frame = 0;
    const updateSize = () => {
      const rect = node.getBoundingClientRect();
      setSize({
        width: Math.max(1, Math.floor(rect.width)),
        height: Math.max(1, Math.floor(rect.height)),
      });
    };

    frame = window.requestAnimationFrame(updateSize);
    const observer = new ResizeObserver(updateSize);
    observer.observe(node);

    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [node]);

  return {
    containerRef,
    width: size.width,
    height: size.height,
    isReady: size.width > 1 && size.height > 1,
  };
}
