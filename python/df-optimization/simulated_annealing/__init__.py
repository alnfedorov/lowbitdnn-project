from .acceptance import metropolis_acceptance_prob, gsa_acceptance_prob
from .engine import simulated_annealing
from .temperature import gsa_schedule, log_schedule, linear_schedule
from .annealing import fast_annealing, generalized_annealing


# So, how does dual annealing looks like?
# 1. Choice of the temperature schedule.
# 2. PDF for data. Variables assumed to be continuous.
# 3. Acceptance probability.

# from scipy.optimize import dual_annealing

# NOTE, that all nice statistical properties about convergence applies ONLY to low dimensional problems.
# NO BODY KNOWS ANYTHING ABOUT EXTREMELY HIGH DIMENSIONAL STUFF. Perhaps, genetic algorithms are better in such a case.
#

# TODO: Realise simulated annealing and generalized dual annealing.
# Everything in pytorch + don't forget about speed. It's important.

# Each of the point MIGHT have multiple parameters. However it's better to model them using inheritance.
# But actual functions should be an arguments.

# Possible choices for 1(k is step):
# Linear: T = T0/k^beta * alpha                                             # DONE
# Log scale: T = T0 / ln(1+k)                                               # DONE
# Non trivial one, that comes from generalized simulated annealing papers.  # DONE

# PDF for data. Usually it't not distribution for X`, but for X` - X:
# Uniform                                                               # DONE
# Integer gaussian                                                      # DONE
# Trick from very fast annealing
# Non trivial one, that comes from generalized simulated annealing papers. Clipped to simple integers
# All the way of the keras initializers

# Acceptance probability:
# Trivial: min(1, np.exp(-delta / T))                       # DONE
# Non trivial from generalized simulated annealing papers.  # DONE
