import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef7ff",
          100: "#d9edff",
          200: "#bce0ff",
          300: "#8ecdff",
          400: "#59b0ff",
          500: "#2f8fff",
          600: "#1670f5",
          700: "#0f59e1",
          800: "#1449b6",
          900: "#16408f",
        },
        // Semantic tokens — values live in globals.css and swap with the theme.
        page: "rgb(var(--c-page) / <alpha-value>)",
        hero: "rgb(var(--c-hero) / <alpha-value>)",
        ink: {
          DEFAULT: "rgb(var(--c-ink1) / <alpha-value>)",
          2: "rgb(var(--c-ink2) / <alpha-value>)",
          3: "rgb(var(--c-ink3) / <alpha-value>)",
          4: "rgb(var(--c-ink4) / <alpha-value>)",
          5: "rgb(var(--c-ink5) / <alpha-value>)",
        },
        ok: "rgb(var(--c-ok) / <alpha-value>)",
        warn: "rgb(var(--c-warn) / <alpha-value>)",
        bad: "rgb(var(--c-bad) / <alpha-value>)",
        surface: {
          DEFAULT: "var(--surface)",
          2: "var(--surface-2)",
          3: "var(--surface-3)",
        },
        line: {
          DEFAULT: "var(--line)",
          2: "var(--line-2)",
        },
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
