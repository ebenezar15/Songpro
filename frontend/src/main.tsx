import { createRoot } from "react-dom/client";
import "@fontsource/catamaran/500.css";
import "@fontsource/catamaran/700.css";
import "@fontsource/catamaran/800.css";
import "@fontsource/hind-madurai/400.css";
import "@fontsource/hind-madurai/500.css";
import "@fontsource/hind-madurai/600.css";
import "./styles.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(<App />);
