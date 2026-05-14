import { Card } from "@/components/ui/card";
import Image from "next/image";

interface StatusMetricCardProps {
  type: "closed" | "yawn";
  events: number;
}

export function StatusMetricCard({ type, events }: StatusMetricCardProps) {
  const isClosed = type === "closed";
  const title = isClosed ? "EYES CLOSED" : "YAWN";
  const imgSrc = isClosed ? "/eye.png" : "/yawn.png";

  const theme = isClosed
    ? {
        title: "text-orange-600",
        number: "text-orange-500",
        iconBg: "bg-orange-50",
        iconRing: "ring-orange-100",
        glow: "bg-orange-200/40",
      }
    : {
        title: "text-rose-600",
        number: "text-rose-500",
        iconBg: "bg-rose-50",
        iconRing: "ring-rose-100",
        glow: "bg-rose-200/40",
      };

  return (
    <Card className="relative h-full overflow-hidden rounded-[2rem] border border-slate-200/70 bg-white shadow-sm transition-all duration-300 hover:shadow-md">
      <div
        className={`pointer-events-none absolute -right-12 -top-12 h-32 w-32 rounded-full blur-2xl ${theme.glow}`}
        aria-hidden
      />

      <div className="relative flex h-full items-center gap-3.5 px-4 py-4 sm:px-5">
        <div
          className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl ring-1 ${theme.iconBg} ${theme.iconRing}`}
        >
          <div className="relative h-9 w-9">
            <Image
              src={imgSrc}
              alt={title}
              fill
              sizes="36px"
              className="object-contain drop-shadow-sm"
              priority
            />
          </div>
        </div>

        <div className="flex h-14 min-w-0 flex-1 flex-col justify-between py-0.5">
          <h3
            className={`whitespace-nowrap text-[0.95rem] font-extrabold uppercase leading-none tracking-[0.05em] ${theme.title}`}
          >
            {title}
          </h3>
          <div className="flex items-baseline gap-1.5">
            <span
              className={`text-[2.75rem] font-extrabold leading-none tracking-tight tabular-nums ${theme.number}`}
            >
              {events}
            </span>
            <span className="text-xs font-medium text-slate-400">events</span>
          </div>
        </div>
      </div>
    </Card>
  );
}
