import { useCallback, useEffect, useState } from "react";

export type Fetcher = (url: string) => Promise<unknown>;

export type Load<T> =
  | { state: "loading" }
  | { state: "error"; message: string; retry: () => void }
  | { state: "ready"; data: T };

/** Module-level so the effect's dependency stays stable across renders. */
export const browserFetch: Fetcher = async (url) => {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url} answered ${r.status}`);
  return r.json();
};

export function useJson<T>(url: string, fetcher: Fetcher = browserFetch): Load<T> {
  const [attempt, setAttempt] = useState(0);
  const [load, setLoad] = useState<Load<T>>({ state: "loading" });
  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  useEffect(() => {
    let live = true;
    setLoad({ state: "loading" });
    fetcher(url)
      .then((d) => live && setLoad({ state: "ready", data: d as T }))
      .catch((e: unknown) => live && setLoad({ state: "error", message: e instanceof Error ? e.message : String(e), retry }));
    return () => {
      live = false;
    };
  }, [url, fetcher, attempt, retry]);
  return load;
}
