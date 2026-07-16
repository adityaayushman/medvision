import type { ReviewStatus } from "./types";

export const REVIEW_STATUS_META: Record<ReviewStatus, { label: string; chip: string }> = {
  pending: { label: "Pending", chip: "chip-info" },
  "in-review": { label: "In review", chip: "chip-warn" },
  reviewed: { label: "Reviewed", chip: "chip-ok" },
  flagged: { label: "Flagged", chip: "chip-bad" },
};

export const REVIEW_STATUSES: ReviewStatus[] = ["pending", "in-review", "reviewed", "flagged"];
