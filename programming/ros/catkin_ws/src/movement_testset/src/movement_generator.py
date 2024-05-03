import random as rand

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class MovementGenerator:
    def __init__(self, duration=60, iterations=100):
        self.inputs = np.linspace(0, np.pi * duration, 500 * duration + 10)
        self.outputs = np.zeros_like(self.inputs)
        self.duration = duration
        self.iterations = iterations

    def sin(self, amplitude, stretch, offset):
        return np.add(
            offset,
            np.multiply(amplitude, np.sin(np.multiply(stretch, self.inputs))),
        )

    def van_der_pol(self, a, b, mu):
        def vdp(t, z):
            x, y = z
            return [y, mu * (1 - x**2) * y - x]

        t = np.linspace(a, b, 500 * self.duration + 10)

        return solve_ivp(vdp, [a, b], [1, 0], t_eval=t).y[0]

    def sigmoid_activation_function(self, offset, amplitude):
        norm_arr = []
        for i in self.outputs:
            temp = ((amplitude - offset) / (1 + np.exp(-i))) + offset
            norm_arr.append(temp)
        return norm_arr

    def layering(self):
        for i in range(self.iterations):
            amplitude = rand.random()
            stretch = rand.random()
            offset = (rand.random() - 0.5) * 2
            np.add(self.outputs, self.sin(amplitude, stretch, offset), out=self.outputs)

    def normalize(self, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(self.outputs) - min(self.outputs)
        for i in self.outputs:
            temp = (((i - min(self.outputs)) * diff) / diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr

    def generate(self):
        # self.layering()
        for i in range(10):
            np.add(
                self.van_der_pol(
                    rand.randint(0, 19),
                    rand.randint(20, 100),
                    rand.randint(0, 20) * 0.5,
                ),
                self.outputs,
                out=self.outputs,
            )

        self.outputs = self.normalize(-0.9, np.pi)
        # return (self.outputs, self.sigmoid_activation_function(-0.9, np.pi))
        return self.outputs


def main():
    mg = MovementGenerator()
    # generated = mg.generate()
    # movements = np.reshape(generated[1], (-1, 1))
    # poss = np.zeros((len(movements), 3))
    # poss = np.insert(poss, [3], movements, axis=1)
    # print(poss)
    # plt.plot(generated[0])
    # plt.plot(generated[1][::10])
    # plt.plot(movements[::10])
    for i in range(10):
        np.add(
            mg.van_der_pol(
                rand.randint(0, 19), rand.randint(20, 100), rand.randint(0, 20) * 0.5
            ),
            mg.outputs,
            out=mg.outputs,
        )
    plt.plot(mg.normalize(-0.9, np.pi))
    # plt.legend([f"$\mu={m}$"])
    plt.show()


if __name__ == "__main__":
    main()
