import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

// On GitHub Actions, GITHUB_REPOSITORY is "user/repo-name".
// GitHub Pages serves at https://user.github.io/repo-name/ so we need that base.
const base = process.env.GITHUB_REPOSITORY
  ? `/${process.env.GITHUB_REPOSITORY.split('/')[1]}/`
  : '/';

export default defineConfig({
  base,
  plugins: [wasm(), topLevelAwait()],
});
