import type { Metadata } from "next";
import { VideoUploadAnalysis } from "@/components/video-upload/VideoUploadAnalysis";

export const metadata: Metadata = {
  title: "Video Upload Analysis | VisionGuard",
  description:
    "Uploaded-video rule-based warning-candidate review workstation.",
};

export default function VideoUploadPage() {
  return <VideoUploadAnalysis />;
}
