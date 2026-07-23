import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

// Self-hosted Inter Variable — humanist sans, all weights in one woff2.
// Bundled with the build, so it works offline (matters for the demo bundle).
import "@fontsource-variable/inter";

import "./index.css";

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("missing #root in index.html");

createRoot(rootEl).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
