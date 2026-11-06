# ACTIVE TASK
Perform a surgical, in-place rename of the product name to "FlowLink" across all project files, keeping all existing code, UI layouts, CSS styles, and backend logic completely unchanged.

# CONSTRAINTS
- Strict Scope: Only update product name strings. Do NOT alter any HTML layout, CSS classes, SVG map generators, JavaScript handlers, or API endpoint routes.
- Target Files to Update:
  - `public/index.html` (or `index.html`)
  - `src/app.py` (and any related `src/*.py` scripts)
  - `CONTEXT.md` and `README.md` (if present)
- Exact Replacements to Perform (case-insensitive where applicable):
  - "SmoothFlow" / "smooth flow" / "Smooth Flow" -> `FlowLink`
  - "Marina Ops" -> `FlowLink`
  - "DunesDay" -> `FlowLink`
- Specific UI Locations in `public/index.html`:
  - Browser tab title: `<title>FlowLink — Chennai Transit Operations</title>`
  - Top navigation bar brand text: replace the logo/title with `FlowLink` (e.g. `FlowLink // CUMTA GRID`)
  - Login / Splash screen header: update main title to `FlowLink`
  - Footer and dialog headers: update any remaining legacy product mentions to `FlowLink`

# DEFINITION OF DONE
- Running `python src/app.py` boots the server cleanly on `http://127.0.0.1:8000`.
- Loading `http://127.0.0.1:8000` shows the exact same working UI with every visible product name displaying **FlowLink**.
- Zero functional or layout regressions in the route cards, timeline, or simulation drawer.