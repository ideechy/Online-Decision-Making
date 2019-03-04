import pstats

p1 = pstats.Stats("profile1.stats")
p1.sort_stats("cumtime")
p1.print_stats(.01)

p2 = pstats.Stats("profile2.stats")
p2.sort_stats("cumtime")
p2.print_stats(.01)
