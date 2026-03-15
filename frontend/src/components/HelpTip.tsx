"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";

interface HelpTipProps {
  /** Short plain-language explanation of the term. */
  text: string;
  /** Optional: size of the ? icon in pixels. Default 14. */
  size?: number;
}

/**
 * A small "?" icon that shows a tooltip on hover (desktop) or tap (mobile).
 *
 * The tooltip is portalled to document.body so it is never clipped by
 * parent overflow:hidden containers (e.g. the app shell content panel).
 * It auto-flips below when near the top and clamps horizontally.
 */
export default function HelpTip({ text, size = 14 }: HelpTipProps) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  const wrapRef = useRef<HTMLSpanElement>(null);
  const [pos, setPos] = useState<{
    top: number;
    left: number;
    arrowLeft: number;
    flipped: boolean;
  } | null>(null);

  // Close on outside click (for mobile tap-to-open)
  const handleOutside = useCallback(
    (e: MouseEvent) => {
      if (
        open &&
        wrapRef.current &&
        !wrapRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    },
    [open],
  );

  useEffect(() => {
    document.addEventListener("mousedown", handleOutside);
    return () => document.removeEventListener("mousedown", handleOutside);
  }, [handleOutside]);

  // Compute position whenever tooltip opens
  useEffect(() => {
    if (!open || !btnRef.current) {
      setPos(null);
      return;
    }

    const compute = () => {
      if (!btnRef.current) return;
      const btn = btnRef.current.getBoundingClientRect();
      const tipW = 256; // w-64 = 16rem = 256px
      const tipH = 60; // approximate; will self-correct on next frame
      const margin = 8;
      const vw = window.innerWidth;

      // Vertical: prefer above, flip below if not enough room
      const flipped = btn.top < tipH + 12;
      const top = flipped
        ? btn.bottom + 8
        : btn.top - tipH - 8;

      // Horizontal: center on button, clamp to viewport
      const btnCenter = btn.left + btn.width / 2;
      let left = btnCenter - tipW / 2;
      if (left < margin) left = margin;
      if (left + tipW > vw - margin) left = vw - margin - tipW;

      const arrowLeft = btnCenter - left;

      setPos({ top, left, arrowLeft, flipped });
    };

    // Compute immediately, then refine after paint (in case tipH estimate was off)
    compute();
    const raf = requestAnimationFrame(compute);
    return () => cancelAnimationFrame(raf);
  }, [open]);

  return (
    <span ref={wrapRef} className="inline-flex items-center ml-1">
      <button
        ref={btnRef}
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

      {open &&
        pos &&
        createPortal(
          <div
            role="tooltip"
            className="fixed z-[9999] w-64 max-w-[90vw] px-3 py-2 rounded-lg
                       bg-gray-800 border border-gray-700 shadow-lg
                       text-xs text-gray-300 leading-relaxed
                       pointer-events-none"
            style={{ top: pos.top, left: pos.left }}
            ref={(el) => {
              // Refine vertical position once we know actual height
              if (!el || !btnRef.current) return;
              const actualH = el.getBoundingClientRect().height;
              const btn = btnRef.current.getBoundingClientRect();
              if (!pos.flipped) {
                el.style.top = `${btn.top - actualH - 8}px`;
              }
            }}
          >
            {text}
            {/* Arrow */}
            <div
              className={`absolute border-4 border-transparent ${
                pos.flipped
                  ? "bottom-full border-b-gray-800"
                  : "top-full border-t-gray-800"
              }`}
              style={{
                left: pos.arrowLeft,
                transform: "translateX(-50%)",
              }}
            />
          </div>,
          document.body,
        )}
    </span>
  );
}
