import type { Metadata } from "next";
import { InsightsPage } from "@/components/insights/InsightsPage";

export const metadata: Metadata = {
  title: "Insights | VisionGuard",
  description: "User-scoped local warning-candidate analytics.",
};

export default function InsightsRoute() {
  return <InsightsPage />;
}
