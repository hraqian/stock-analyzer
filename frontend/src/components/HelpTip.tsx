"use client";

import { useState, useRef, useEffect, useCallback } from "react";

interface HelpTipProps {
  /** Short plain-language explanation of the term. */
  text: string;
  /** Optional: size of the ? icon in pixels. Default 14. */
  size?: number;
}

/**
 * A small "?" icon that shows a tooltip on hover (desktop) or tap (mobile).
 *
 * Usage:
 *   <span>RSI <HelpTip text="Relative Strength Index measures..." /></span>
 */
export default function HelpTip({ text, size = 14 }: HelpTipProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);

  // Close on outside click (for mobile tap-to-open)
  const handleOutside = useCallback(
    (e: MouseEvent) => {
      if (
        open &&
        ref.current &&
        !ref.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    },
    [open]
  );

  useEffect(() => {
    document.addEventListener("mousedown", handleOutside);
    return () => document.removeEventListener("mousedown", handleOutside);
  }, [handleOutside]);

  // Reposition tooltip if it overflows viewport
  useEffect(() => {
    if (!open || !tipRef.current) return;
    const rect = tipRef.current.getBoundingClientRect();
    // If overflowing right edge, shift left
    if (rect.right > window.innerWidth - 8) {
      tipRef.current.style.left = "auto";
      tipRef.current.style.right = "0";
    }
    // If overflowing left edge, shift right
    if (rect.left < 8) {
      tipRef.current.style.left = "0";
      tipRef.current.style.right = "auto";
    }
  }, [open]);

  return (
    <span ref={ref} className="relative inline-flex items-center ml-1">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        aria-label="Help"
        className="inline-flex items-center justify-center rounded-full
                   bg-gray-700 hover:bg-gray-600 text-gray-400 hover:text-gray-200
                   transition-colors cursor-help focus:outline-none
                   focus-visible:ring-1 focus-visible:ring-blue-500"
        style={{ width: size, height: size, fontSize: size * 0.65 }}
      >
        ?
      </button>

      {open && (
        <div
          ref={tipRef}
          role="tooltip"
          className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2
                     w-64 max-w-[90vw] px-3 py-2 rounded-lg
                     bg-gray-800 border border-gray-700 shadow-lg
                     text-xs text-gray-300 leading-relaxed
                     pointer-events-none"
        >
          {text}
          {/* Arrow */}
          <div
            className="absolute top-full left-1/2 -translate-x-1/2
                       border-4 border-transparent border-t-gray-800"
          />
        </div>
      )}
    </span>
  );
}
