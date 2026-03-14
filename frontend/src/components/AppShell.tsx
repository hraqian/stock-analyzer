"use client";

import { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";

function AppContent({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const pathname = usePathname();

  // Show loading spinner while checking auth
  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-950">
        <div className="text-gray-500 text-sm">Loading...</div>
      </div>
    );
  }

  // Login page gets no shell
  if (pathname === "/login") {
    return <>{children}</>;
  }

  // Not logged in — redirect to login
  if (!user) {
    // Use a client-side redirect
    if (typeof window !== "undefined") {
      window.location.href = "/login";
    }
    return null;
  }

  // Authenticated layout: sidebar + header + content
  return (
    <div className="h-screen flex overflow-hidden bg-gray-950">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <AuthProvider>
      <AppContent>{children}</AppContent>
    </AuthProvider>
  );
}
