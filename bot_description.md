# PiG_Bot

**A StarCraft II training opponent that models its opponent — and remembers them.**

Most bots execute a build script against whatever you do. PiG_Bot treats every game
as a probability problem: it scouts to gather evidence, runs the observations through
trained models, and acts on the resulting beliefs.

## Machine learning under the hood

The belief layer is built on Bayesian principles — priors over what the opponent is
doing, updated by scouting evidence, with uncertainty carried forward instead of
guessed away.

- **Strategy classification** — a gradient-boosted classifier trained on 2,000+ games
  of the bot's own ladder telemetry feeds a belief layer estimating P(cheese), P(all-in),
  P(timing attack), and P(macro) — down to build-level labels like 12-pool or cannon
  rush. Every scouting observation sharpens the posterior.
- **Cross-game opponent modeling** — Dirichlet priors, the classic Bayesian treatment
  of "what does this player tend to do," accumulate your tendencies across matches.
  Beat it with a 12-pool last week? It expects it this week.
- **Probabilistic fog-of-war reasoning** — unseen enemy units carry existence
  probabilities that decay over time. Stale intel fades out; it never flips between
  "fresh" and "useless."
- **Value-of-information scouting** — when blind and behind, it ranks scout
  destinations by staleness × decision impact instead of cycling expansions.

## Trained on its own games

Every ladder game streams telemetry into a replay-grade corpus. Models retrain on
the accumulated evidence, and each new model-epoch ships with priors grounded in
what actually happened. The bot improves through data, not patch notes.

## Fundamentals

Macro to the Bronze-to-GM standard, scouting before committing, and cheese reaction
builds for the classics (Zergling rush, cannon rush, proxy rax). Adjustable
difficulty is in active development.

Built with the ARES framework and python-sc2. Open source:
https://github.com/Vers-AI/SC2_PiGBot
