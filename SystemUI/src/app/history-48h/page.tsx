import type { Metadata } from "next";
import { History48hPage } from "@/components/history-48h/History48hPage";

export const metadata: Metadata = {
  title: "48h History | VisionGuard",
  description: "Recent driver-state warning-candidate history.",
};

export default function History48hRoute() {
  return <History48hPage />;
}
