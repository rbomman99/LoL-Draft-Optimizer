# League of Legends Draft Optimizer

An application in development that recommends champion picks during League of Legends
champion select. The goal is to help players choose champions that fit their team,
match up against the enemy composition, and suit their role and rank.

## Implemented

- **Match collection:** Fetch ranked solo matches from the Riot Games API, starting
  with EUW Challenger players. Store champion picks, team roles, outcomes, patch,
  and timestamps in PostgreSQL. Retry rate-limited requests and skip stored matches
  so collection can continue after an interruption.
- **Training workflow:** Train a PyTorch win/loss model on complete compositions and
  versions with some picks hidden to approximate unfinished drafts. Split matches
  chronologically before creating these examples, validate on the newest 20%, and
  save the dataset snapshot, evaluation metrics, and best model checkpoint.
- **Draft recommendations:** Accept allied picks, enemy picks, bans, role, and team
  side. Add each eligible candidate to the draft, score the resulting composition,
  and return ranked picks. Exclude unavailable champions and require a minimum
  number of recorded games in the requested role.
- **API access:** Provide endpoints to request recommendations, list supported
  champion IDs and their role history, and check model readiness. Requests can be
  tried through FastAPI's interactive API documentation.
- **Local services:** Run PostgreSQL, collection/training workers, and the API with
  Docker. Keep database records in a persistent volume and model artifacts on disk.

These features form an experimental backend. Recommendation quality has not yet
been established on a large dataset, and model scores are not proven win-rate
improvements. Current role support filters candidates; the model does not encode
role assignments or player rank. Partial drafts are simulated from completed games,
not reconstructed from historical pick order.

## Planned

- **Champion-select interface:** Build a React and TypeScript page with champion
  names and portraits, search, allied/enemy pick slots, bans, role selection, and
  ranked recommendations that refresh as the draft changes.
- **Rank-specific recommendations:** Collect participant rank information and add
  rank-aware training so users can request suggestions for their level of play.
- **Useful explanations:** Investigate measurable synergy and matchup statistics
  to explain recommendations beyond the current model score and role-game count.
- **Live client integration:** Explore reading champion-select state automatically
  instead of requiring users to enter every pick and ban.
- **Deployment:** Host the frontend, API, and database on AWS with a workflow for
  updating match data and deploying newly evaluated models.
