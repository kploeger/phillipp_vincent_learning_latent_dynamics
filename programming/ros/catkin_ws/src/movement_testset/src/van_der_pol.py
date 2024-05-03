import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def main():
    def vdp(t, z):
        x, y = z
        return [y, mu * (1 - x**2) * y - x]

    a, b = 0, 20

    mus = [1, 2, 3, 4, 5, 0.5, 0.25, 0.01]
    styles = ["-", "--", ":", "-.", ".", "_", "+", "*"]
    t = np.linspace(a, b, 500)

    for mu, style in zip(mus, styles):
        sol = solve_ivp(vdp, [a, b], [1, 0], t_eval=t)
        plt.plot(sol.t, sol.y[0], style)
        # plt.plot(sol.y[0], sol.y[1], style)

    # make a little extra horizontal room for legend
    # plt.xlim([-3, 3])
    plt.legend([f"$\mu={m}$" for m in mus])
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
