"use client";

import { useState, useEffect } from "react";
import { getSheetLinks } from "@/lib/api";
import type { CalloutLink } from "@/types/api";

export function useSheetLinks(sheetId: string | null) {
  const [links, setLinks] = useState<CalloutLink[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!sheetId) {
      setLinks([]);
      return;
    }
    setIsLoading(true);
    getSheetLinks(sheetId)
      .then(setLinks)
      .catch(() => setLinks([]))
      .finally(() => setIsLoading(false));
  }, [sheetId]);

  return { links, isLoading };
}
