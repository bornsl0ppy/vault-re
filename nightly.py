"""One row per night of a once-a-day vault: its exposure going in and the exposure its 01:45 UTC order aims for.
The aim comes from the order's origSz (historicalOrders), so an order that rests and fills hours later still
counts on its night. Exposure is position x BTC / account value, both just before 01:45. Read-only.

    python nightly.py convexity              # -> data/convexity/nightly.json
"""
import datetime as dt
import json
import sys

import numpy as np

from book import Book

DAY = 86_400_000
DECISION = 105 * 60_000                             # 01:45 UTC
WINDOW = 5 * 60_000                                 # the bot places at 01:45:06-10


def table(name):
    book = Book(name)
    placed = {}
    for row in json.load(open(f"data/{name}/orders.json")):
        o = row["order"]
        size = float(o["origSz"]) * (1 if o["side"] == "B" else -1)
        placed[o["oid"]] = (min(o["timestamp"], placed.get(o["oid"], (o["timestamp"],))[0]), size)
    rows = []
    for day in range((book.created // DAY + 1) * DAY, int(book.close_t[-1]) - DECISION + 1, DAY):
        t = day + DECISION
        night = [size for at, size in placed.values() if t <= at < t + WINDOW]
        assert len(night) <= 1, f"{len(night)} orders on {dt.date.fromtimestamp(day / 1000)}"
        order = night[0] if night else 0.0
        pos, pnl, av = (float(x) for x in book.at(t))
        px = float(book.mark(t))
        rows.append(dict(day=str(dt.datetime.fromtimestamp(day / 1000, dt.timezone.utc).date()), t=t, px=px, av=av,
                         pnl=pnl, pos=pos, order=order, prev=pos * px / av, y=(pos + order) * px / av))
    return rows


if __name__ == "__main__":
    name = sys.argv[1]
    rows = table(name)
    json.dump(rows, open(f"data/{name}/nightly.json", "w"))
    y = np.array([r["y"] for r in rows])
    print(f"{len(rows)} nights {rows[0]['day']} to {rows[-1]['day']}; {sum(r['order'] != 0 for r in rows)} with a "
          f"01:45 order; |exposure| > 1x on {(np.abs(y) > 1).sum()}")
