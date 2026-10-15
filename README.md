# Matchue 2048

Two models play 2048 through [OpenRouter](https://openrouter.ai), and every move is logged with its latency and cost.

- **GPT-6 Luna** (`openai/gpt-6-luna`), a chat model that reasons before it answers.
- **Jev** (`typesafe/jev-1.13`), a decision model. It gets the board and the legal moves and returns one move with a probability for each option. It doesn't reason step by step.

There is also a random player as a free baseline.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
```

## Usage

```bash
python main.py run --players gpt jev random --games 20
python main.py resume runs/<run>              # continue after a crash or Ctrl+C
python main.py resume runs/<run> --games 30   # add more games to a run
python main.py summarize runs/<run>
```

`run` also takes `--max-moves`, `--reasoning-effort` (default `low`), `--gpt-model`, `--jev-model` and `--name`. All games in a run play at the same time.

## How It Works

Both models get the same rules and see the same thing on each move: the board, the score, the move number, the number of empty cells and the legal moves. They don't see earlier moves, and they can only answer with a legal move. When just one move is legal, it is played without asking the model.

Game N uses seed N for its tile spawns, so every player starts game N from the same board.

Costs are the `usage.cost` values OpenRouter returns with each response.

## Output

Each run writes to `runs/<timestamp>-<name>/`:

- `run.json`: settings, prompts and OpenRouter pricing for each model.
- `games/<player>/seed-NNNN.jsonl`: one line per move with the board before and after, where each tile moved, the new tile, the score, the model's answer (GPT's one-sentence reason and reasoning summary, Jev's move probabilities), latency, tokens, cost and running totals.
- `games/<player>/seed-NNNN.summary.json`: the result of each game, its total cost and time, and the move where each tile value first appeared.
- `raw/<player>/seed-NNNN.jsonl`: every API request and response, including retries.
- `summary.json` and `summary.md`: per-player statistics and head-to-head results on shared seeds.
- `events.log`: a line per move.

Boards are rows from top to bottom, with 0 for an empty cell. Positions are `[row, col]` from the top-left.

## License

MIT
