# ACTIVE TASK
Sanitize all hardcoded API keys/secrets across the repository, move the key into `.env`, add `.env` to `.gitignore`, amend the latest git commit to erase secrets from history, and push cleanly to GitHub.

# CONSTRAINTS
1. Search and Locate Secrets:
   - Search across all repository files for sensitive tokens: `sk-or-`, `AIza`, `OPENROUTER_API_KEY`, and raw authorization tokens.
   - Print the files and line numbers where any exposed secrets were detected.

2. Move Secret to Environment Variable:
   - If a raw API key is found hardcoded in any `.py`, `.json`, `.js`, or `.html` file:
     - Extract it and append it to `.env` as `OPENROUTER_API_KEY=...` (create `.env` if missing).
     - Replace the hardcoded string in the source code with `os.getenv("OPENROUTER_API_KEY", "")` (in Python) or ensure frontend endpoints fetch it strictly from the backend server without exposing it in client-side code.
   - Verify `.env` is listed inside `.gitignore`. If missing, append `.env` to `.gitignore`.

3. Scrub Git History for Clean Push:
   - Run `git rm --cached .env` (if it was previously tracked).
   - Stage all sanitized files and updated `.gitignore`: `git add -A`.
   - Amend the latest commit to eliminate the secret from git history:
     `git commit --amend --no-edit`
   - If the secret exists across multiple unpushed local commits, rewrite them cleanly with `git reset --soft origin/main` (or origin/master), re-stage, and commit with:
     `git commit -m "DunesDay: Secure transit operations MVP without exposed secrets"`

4. Push to GitHub:
   - Attempt `git push origin HEAD`.
   - If GitHub displays an unblock/bypass URL in stderr, report the URL in the terminal so it can be unblocked if needed.

# DEFINITION OF DONE
- No raw `sk-or-` or `AIza` secrets remain hardcoded in any tracked project files.
- `.gitignore` contains `.env`.
- `git status` shows a clean working tree.
- `git push` succeeds or provides the direct GitHub secret unblock URL.