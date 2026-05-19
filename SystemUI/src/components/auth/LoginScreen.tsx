"use client";

import { useState, type FormEvent } from "react";
import { Eye, LockKeyhole, ShieldCheck, UserRound } from "lucide-react";
import {
  VISION_GUARD_LOCAL_ACCOUNT_USERNAME,
  useVisionGuardAuth,
} from "@/lib/authStore";

export function LoginScreen() {
  const { loginWithPassword } = useVisionGuardAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const ok = loginWithPassword({ username, password });

    if (!ok) {
      setError("Username or password is incorrect for this local MVP account.");
      setPassword("");
      return;
    }

    setError("");
  }

  return (
    <main className="flex min-h-dvh items-center justify-center bg-[radial-gradient(circle_at_top_left,_rgba(37,99,235,0.14),_transparent_32%),linear-gradient(135deg,_#f8fafc_0%,_#eef4f8_52%,_#e5edf5_100%)] px-4 py-8 text-slate-950 dark:bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.16),_transparent_32%),linear-gradient(135deg,_#020617_0%,_#0f172a_56%,_#111827_100%)] dark:text-slate-50">
      <div className="grid w-full max-w-5xl gap-5 lg:grid-cols-[0.95fr_1.05fr]">
        <section className="flex min-h-[520px] flex-col justify-between rounded-3xl border border-white/70 bg-white/80 p-7 shadow-2xl shadow-slate-950/10 backdrop-blur-xl dark:border-slate-700/70 dark:bg-slate-950/70 dark:shadow-black/30">
          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-600 text-white shadow-lg shadow-blue-600/25">
                <Eye className="h-6 w-6" strokeWidth={2.4} />
              </div>
              <div>
                <h1 className="text-2xl font-black tracking-tight">
                  VisionGuard
                </h1>
                <p className="text-sm font-semibold text-slate-500 dark:text-slate-400">
                  Driver Drowsiness System
                </p>
              </div>
            </div>

            <div className="mt-10">
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-blue-600 dark:text-cyan-300">
                Local account foundation
              </p>
              <h2 className="mt-3 text-4xl font-black tracking-tight">
                Sign in to continue monitoring.
              </h2>
              <p className="mt-4 max-w-lg text-sm leading-6 text-slate-600 dark:text-slate-300">
                This system provides rule-based warning-candidate analysis, not
                final system-level drowsiness accuracy.
              </p>
            </div>
          </div>

          <div className="rounded-2xl border border-blue-100 bg-blue-50/80 p-4 text-sm leading-6 text-blue-900 dark:border-cyan-400/20 dark:bg-cyan-400/10 dark:text-cyan-100">
            Local MVP account. This browser-only password check is not
            production authentication; server-side authentication and persisted
            user-owned history are future work.
          </div>
        </section>

        <section className="rounded-3xl border border-white/80 bg-white p-6 shadow-2xl shadow-slate-950/10 dark:border-slate-700/70 dark:bg-slate-900 dark:shadow-black/30">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-100 text-blue-600 dark:bg-slate-800 dark:text-cyan-300">
              <ShieldCheck className="h-5 w-5" strokeWidth={2.3} />
            </div>
            <div>
              <h2 className="text-xl font-black tracking-tight">
                Local login
              </h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Use the assigned local MVP username and password.
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="mt-6 space-y-4">
            <label className="block">
              <span className="text-sm font-bold text-slate-800 dark:text-slate-100">
                User name
              </span>
              <span className="mt-2 flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2.5 transition focus-within:border-blue-300 focus-within:ring-2 focus-within:ring-blue-400/30 dark:border-slate-700 dark:bg-slate-950">
                <UserRound className="h-4 w-4 text-slate-400" />
                <input
                  value={username}
                  onChange={(event) => {
                    setUsername(event.target.value);
                    setError("");
                  }}
                  autoComplete="username"
                  className="min-w-0 flex-1 bg-transparent text-sm font-semibold text-slate-900 outline-none placeholder:text-slate-400 dark:text-slate-50"
                  placeholder={VISION_GUARD_LOCAL_ACCOUNT_USERNAME}
                />
              </span>
            </label>

            <label className="block">
              <span className="text-sm font-bold text-slate-800 dark:text-slate-100">
                Password
              </span>
              <span className="mt-2 flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2.5 transition focus-within:border-blue-300 focus-within:ring-2 focus-within:ring-blue-400/30 dark:border-slate-700 dark:bg-slate-950">
                <LockKeyhole className="h-4 w-4 text-slate-400" />
                <input
                  value={password}
                  onChange={(event) => {
                    setPassword(event.target.value);
                    setError("");
                  }}
                  autoComplete="current-password"
                  className="min-w-0 flex-1 bg-transparent text-sm font-semibold text-slate-900 outline-none placeholder:text-slate-400 dark:text-slate-50"
                  placeholder="Password"
                  type="password"
                />
              </span>
            </label>

            {error && (
              <p className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-700 dark:border-rose-400/30 dark:bg-rose-500/10 dark:text-rose-200">
                {error}
              </p>
            )}

            <button
              type="submit"
              className="flex w-full items-center justify-center rounded-2xl bg-blue-600 px-4 py-3 text-sm font-black text-white shadow-lg shadow-blue-600/20 outline-none transition hover:bg-blue-700 focus-visible:ring-2 focus-visible:ring-blue-400"
            >
              Continue
            </button>
          </form>

          <p className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400">
            Only the assigned local account can enter this MVP app shell.
          </p>
        </section>
      </div>
    </main>
  );
}
