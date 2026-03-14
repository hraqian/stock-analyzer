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
 * Auto-flips below when too close to the top of the viewport.
 * Auto-shifts horizontally so the tooltip never hides behind the sidebar or
 * off the right edge.
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

  // Position the tooltip after it renders so it never overflows
  useEffect(() => {
    if (!open || !tipRef.current || !ref.current) return;

    const btn = ref.current.getBoundingClientRect();
    const tip = tipRef.current;
    const tipRect = tip.getBoundingClientRect();
    const vw = window.innerWidth;
    const margin = 8;

    // Vertical: flip below if not enough room above
    if (btn.top < tipRect.height + 12) {
      tip.style.top = `${btn.height + 8}px`;
      tip.style.bottom = "auto";
      tip.dataset.flipped = "true";
    } else {
      tip.style.bottom = `${btn.height + 8}px`;
      tip.style.top = "auto";
      tip.dataset.flipped = "false";
    }

    // Horizontal: start centered, then clamp to viewport
    const btnCenter = btn.left + btn.width / 2;
    let tipLeft = btnCenter - tipRect.width / 2;

    // Don't go past left edge of viewport (accounts for sidebar)
    if (tipLeft < margin) {
      tipLeft = margin;
    }
    // Don't go past right edge
    if (tipLeft + tipRect.width > vw - margin) {
      tipLeft = vw - margin - tipRect.width;
    }

    // Convert from viewport coords to coords relative to the parent span
    const parentLeft = btn.left;
    tip.style.left = `${tipLeft - parentLeft}px`;
    tip.style.transform = "none";

    // Position the arrow to point at the button center
    const arrow = tip.querySelector<HTMLElement>("[data-arrow]");
    if (arrow) {
      const arrowX = btnCenter - tipLeft;
      arrow.style.left = `${arrowX}px`;
      arrow.style.transform = "translateX(-50%)";
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
          className="absolute z-50 w-64 max-w-[90vw] px-3 py-2 rounded-lg
                     bg-gray-800 border border-gray-700 shadow-lg
                     text-xs text-gray-300 leading-relaxed
                     pointer-events-none"
          style={{ left: "-9999px" }}
        >
          {text}
          {/* Arrow — positioned dynamically by the useEffect */}
          <div
            data-arrow
            className="absolute border-4 border-transparent"
            style={{ left: "50%" }}
            ref={(el) => {
              if (!el || !tipRef.current) return;
              const flipped = tipRef.current.dataset.flipped === "true";
              if (flipped) {
                el.style.bottom = "100%";
                el.style.top = "auto";
                el.className =
                  "absolute border-4 border-transparent border-b-gray-800";
              } else {
                el.style.top = "100%";
                el.style.bottom = "auto";
                el.className =
                  "absolute border-4 border-transparent border-t-gray-800";
              }
            }}
          />
        </div>
      )}
    </span>
  );
}
