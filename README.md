# davidegualano.com

One-page site for Davide Gualano, football recruitment analysis (GitHub Pages, custom domain in `CNAME`).

- `index.html`, `style.css`, `script.js`: the whole site. No build step.
- `404.html`: any old link (the notebook folders, `work-with-me.html`) lands on the home page.
- The brief form opens the visitor's email app with the answers filled in. To receive submissions without email,
  swap it for Formspree: set the form's `action` to the Formspree URL, `method="POST"`, and remove the submit handler
  in `script.js`.

On 15 December, when "Which transfers work? Take three" is out, swap every `transfer-success-rates-by-league-v2` link in `index.html` for `https://the-cutback.beehiiv.com/p/which-transfers-work-take-three`. The hero's 43/35/22 split comes from that piece.
