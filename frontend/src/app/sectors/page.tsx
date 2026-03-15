import { redirect } from "next/navigation";

/**
 * Server-side redirect /sectors → /analysis?tab=sectors for backward compatibility.
 */
export default function SectorsRedirect() {
  redirect("/analysis?tab=sectors");
}
