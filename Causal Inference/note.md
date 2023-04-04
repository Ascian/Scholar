# Potential Outcomes

## Assumption

* Unconfoundedness: The treatment assignment(given samples) is independent of the outcome, given the covariates.
  $(Y(1) - Y(0)) \!\perp\!\!\!\perp T | X$
* Positivity：0 < $P(T=1|X)$ < 1, so that $E[Y|T=1, X]$ can be counted
  There would be a positive violation in the rest of the overlapped part between P(X|T=1) and P(X|T=0). The area of the overlapped part would decrease along with the increase of covatiates.
* No interference: $Y_i(t_1, ..., t_i, ..., t_n) = Y_i(t_i)$, means Y is only affected by the treatment of the current sample.
* Consistency: T = t $\implies$ $Y = Y(t)$, means Y(t) is not affected by the different kinds of T.

## Adjust formula

$$ 
\begin{array}{ll} 
E[Y(1) - Y(0)] (No\space interference)= E[Y(1)] - E[Y(0)] \\
= E_xE[Y(1)|X] - E[Y(0)|X] \\
= E_xE[Y(1)|T=1, X] - E[Y(0)|T=0, X] (Unconfoundedness\space  and\space  positivity)\\
= E_x[Y|T=1, X] - E[Y|T=0, X] (consistency)
\end{array}
$$