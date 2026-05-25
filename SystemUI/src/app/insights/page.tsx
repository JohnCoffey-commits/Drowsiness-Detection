import type { Metadata } from "next";
import { InsightsPage } from "@/components/insights/InsightsPage";

export const metadata: Metadata = {
  title: "Insights | VisionGuard",
  description: "Patterns from recent Live Monitor alerts.",
};

export default function InsightsRoute() {
  return <InsightsPage />;
}
