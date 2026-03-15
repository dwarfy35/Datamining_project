## Problems encountered

- Some rows are missing teamid, but teamname is consistently present.

Clustering the teams gives a strong bias towards winning and losing games.
> Solution: Treat them separately

Clustering the teams gives a strong bias towards gamelength.
> Solution: Learn the usual mapping between gamelength and time dependent columns and substract the bias.