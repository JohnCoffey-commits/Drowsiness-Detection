import type { Metadata } from "next";
import { Suspense } from "react";
import { History48hPage } from "@/components/history-48h/History48hPage";

export const metadata: Metadata = {
  title: "History | VisionGuard",
  description: "Recent Live Monitor driving alert history.",
};

export default function History48hRoute() {
  return (
    <Suspense fallback={null}>
      <History48hPage />
    </Suspense>
  );
}
