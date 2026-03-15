/**
 * Shared spinning loading indicator.
 *
 * Replaces the 4+ duplicated spinner SVGs across strategy, scanner, etc.
 */

interface SpinnerProps {
  /** Tailwind size classes (default: "h-4 w-4") */
  size?: string;
  /** Additional CSS classes */
  className?: string;
}

export default function Spinner({ size = "h-4 w-4", className = "" }: SpinnerProps) {
  return (
    <svg
      className={`animate-spin ${size} ${className}`}
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}
