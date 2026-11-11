# davidegualano.com

Site for Davide Gualano, football recruitment analysis (GitHub Pages, custom domain in `CNAME`). No build step.

- `index.html`: the home page, for clubs, agencies and editors. `method.html`: the models, for analysts and hiring teams.
- `style.css`, `script.js`: shared by both pages. `fonts/`: Archivo and Source Sans 3 (SIL Open Font License), served
  from the site, not from Google.
- `images/share.png`: the picture LinkedIn / X show when the site is linked. `favicon.svg`, `images/icon-180.png`: icons.
- `404.html`: any old link lands on the home page.

## Things that update themselves
- **Price lookup** ("What would he cost? Try it."): reads `data/players.csv` from the price app's repo
  (`Gibranium/what-would-he-cost`, branch `main`) every time a visitor uses it. Refresh the data and push the app as
  usual: the site follows, no push here. It needs the columns `player, team, league, role, age, minutes, season,
  worth, worth_low, worth_high, price, price_low, price_high` and the season label `2025/26` (`SEASON` in `script.js`:
  change it when the app's main season changes). If the file can't be read, the section shows a link to the app.

## Things to change by hand
- **Brief form**: sent from the page to Formspree (form `xjywvoqq`), which forwards it by email. The URL is on the
  `<form>` in `index.html`, twice: `action` (used if the visitor's JavaScript is off) and `data-endpoint` (used
  otherwise). If sending fails, the visitor's email app opens with the answers filled in.
- **Hero numbers** (13,765 moves, 43 / 35 / 22): in the hero of `index.html`, in the two shares at the top of the
  dots block in `script.js` (0.428, 0.347), and in `images/share.png`.
- **Prices** (€200 / €600 / from €400): hero line, the three offers, the lookup button text in `script.js`, and the
  media line.
- **On 15 December**, when "Which transfers work? Take three" is out: swap every `transfer-success-rates-by-league-v2`
  link (`index.html`, `method.html`) for `https://the-cutback.beehiiv.com/p/which-transfers-work-take-three`, and
  reword the "Which moves work" card in Proof and the first item on `method.html` (they say "out in December").
