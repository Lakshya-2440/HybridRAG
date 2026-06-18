import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        ink: "#1f2328",
        paper: "#f7f5f1",
        line: "#d8d2c7",
        fern: "#18745c",
        clay: "#9c523f",
        honey: "#b7791f",
      },
    },
  },
  plugins: [],
};

export default config;
