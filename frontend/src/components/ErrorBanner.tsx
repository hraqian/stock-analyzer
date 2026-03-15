/**
 * Shared error banner component.
 *
 * Replaces the 7+ duplicated error display blocks across the app.
 */

interface ErrorBannerProps {
  /** Error message to display */
  message: string | null;
  /** Optional dismiss callback (shows an "x" button) */
  onDismiss?: () => void;
  /** Optional variant for amber warnings vs red errors */
  variant?: "error" | "warning";
}

export default function ErrorBanner({ message, onDismiss, variant = "error" }: ErrorBannerProps) {
  if (!message) return null;

  const colors =
    variant === "warning"
      ? "bg-amber-900/20 border-amber-800 text-amber-300"
      : "bg-red-900/30 border-red-800 text-red-300";

  return (
    <div className={`${colors} border rounded-lg p-4 text-sm flex items-start gap-2`}>
      <span className="flex-1">{message}</span>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-red-400 hover:text-red-300 text-xs shrink-0"
        >
          dismiss
        </button>
      )}
    </div>
  );
}
