"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * Redirect /sectors to /analysis?tab=sectors for backward compatibility.
 */
export default function SectorsRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/analysis?tab=sectors");
  }, [router]);
  return null;
}
