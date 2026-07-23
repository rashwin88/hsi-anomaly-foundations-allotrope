/// <reference types="vite/client" />

// @fontsource-variable/inter ships CSS only (no .d.ts). Declaring the module
// lets TypeScript accept the side-effect import in main.tsx.
declare module "@fontsource-variable/inter";
