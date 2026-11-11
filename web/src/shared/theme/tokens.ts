/** Design tokens. Components read these; no colour, size or duration is written anywhere else.
 * The palette is a delivery bag's: a cool off-white surface, near-black ink, basil for what the new page
 * does, saffron for a kitchen that is getting full, and chili for what went wrong. */
export const color = {
  surface: "#F1F3EF",
  card: "#FFFFFF",
  ink: "#171A17",
  muted: "#5A615A",
  line: "#D3D8D1",
  panel: "#E2E7E0",
  basil: "#2C7A4B",
  basilSoft: "#DCEDE2",
  saffron: "#D99A12",
  saffronSoft: "#F8EBCB",
  chili: "#B83224",
  chiliSoft: "#F5DCD8",
  focus: "#1F5FBF",
} as const;

export const font = {
  display: '"Bricolage Grotesque", system-ui, sans-serif',
  body: '"Figtree", system-ui, sans-serif',
} as const;

/** 1.25 type scale from 16 px. */
export const size = { xs: 12.8, sm: 14, md: 16, lg: 20, xl: 25, xxl: 31.25, display: 52 } as const;
export const weight = { regular: 400, medium: 500, semibold: 600, bold: 700, heavy: 800 } as const;
export const leading = { tight: 1.05, snug: 1.25, body: 1.55 } as const;
export const space = [0, 4, 8, 12, 16, 24, 32, 48, 64, 96] as const;
export const radius = { sm: 6, md: 12, lg: 22, phone: 34, pill: 999 } as const;
export const measure = { text: 66, page: 1200 } as const;
export const phone = { width: 300, cardWidth: 112, cardHeight: 92 } as const;
export const stroke = { hair: 1, rule: 2 } as const;
export const motion = {
  fast: 0.18,
  base: 0.35,
  stagger: 0.04,
  spring: { type: "spring", stiffness: 260, damping: 30 } as const,
  ease: [0.22, 1, 0.36, 1] as [number, number, number, number],
} as const;
export const breakpoint = { narrow: 820 } as const;
